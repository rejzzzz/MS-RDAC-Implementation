import os, torch, contextlib, numpy as np, time
from tempfile import mkstemp
from .io_utils import read_bitstring, filesize, to_cuda
from .arithmetic_coder import FlatFrequencyTable, SimpleFrequencyTable, ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream
from .ppm_model import PpmModel

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

class BasicEntropyCoder:
    def __init__(self, q_step=64, num_kp=10):
        self.q_step, self.num_kp, self.kp_reference = q_step, num_kp, None

    def q(self, tgt):
        return torch.round((tgt+1)*self.q_step)

    def dq(self, tgt):
        return tgt/self.q_step - 1

    def enc_kp(self, kp_src, kp_drv):
        self.kp_reference = self.kp_reference or kp_src
        sr = torch.round((self.kp_reference['value']+1)*self.q_step)
        dr = torch.round((kp_drv['value']+1)*self.q_step)
        info = self.cmp(sr-dr)
        kp_dec = {'value': to_cuda((sr - info['res'])/self.q_step - 1)}
        self.kp_reference = kp_dec
        return {**info, 'kp_dec': kp_dec}

    def cmp(self, kp):
        shape = kp.shape
        kp = kp.flatten().numpy().astype(np.int8)
        tmp, tmp_path = mkstemp("inp_temp.bin")
        with open(tmp_path, "wb") as f:
            f.write(np.array(kp).tobytes())
        
        freqs = SimpleFrequencyTable(FlatFrequencyTable(257))
        tmp_out, tmp_out_path = mkstemp("out_temp.bin")
        with open(tmp_path, 'rb') as inp, contextlib.closing(BitOutputStream(open(tmp_out_path, "wb"))) as bitout:
            enc = ArithmeticEncoder(32, bitout)
            while True:
                sym = inp.read(1)
                if not sym:
                    break
                enc.write(freqs, sym[0])
                freqs.increment(sym[0])
            enc.write(freqs, 256)
            enc.finish()
        
        bit_size = filesize(tmp_out_path)
        res = np.reshape(self.dcmp(tmp_out_path), shape)
        for p in [tmp, tmp_out]:
            os.close(p)
        os.remove(tmp_path)
        os.remove(tmp_out_path)
        return {'bitstring': read_bitstring(tmp_out_path), 'bitstring_size': bit_size, 'res': res}

    def dcmp(self, in_path):
        dec_p, dec_path = mkstemp("decoding.bin")
        freqs = SimpleFrequencyTable(FlatFrequencyTable(257))
        with open(in_path, "rb") as inp, open(dec_path, "wb") as out:
            dec = ArithmeticDecoder(32, BitInputStream(inp))
            while True:
                sym = dec.read(freqs)
                if sym == 256:
                    break
                out.write(bytes((sym,)))
                freqs.increment(sym)
        
        with open(dec_path, 'rb') as f:
            res = np.frombuffer(f.read(), dtype=np.int8)
        os.close(dec_p)
        os.remove(dec_path)
        return res

class KpEntropyCoder(BasicEntropyCoder):
    def __init__(self, q_step=512, num_kp=10, model_order=0, device='cpu'):
        super().__init__(q_step, num_kp)
        self.hist, self.dec_hist = [], []
        self.ppm = PpmModel(model_order, 257, 256)
        self.dec_ppm = PpmModel(model_order, 257, 256)
        
    def enc_kp(self, kp_tgt, device='cpu'):
        sr = torch.round((self.kp_reference['value']+1.0)*self.q_step)
        dr = torch.round((kp_tgt['value']+1.0)*self.q_step)
        info = self.cmp(sr-dr)
        res_hat = info['res'].cuda() if torch.cuda.is_available() else info['res']
        kp_dec = {'value': (sr - res_hat)/self.q_step - 1.0}
        self.kp_reference = {'value': kp_dec['value'].detach().clone()}
        return {**info, 'kp_hat': kp_dec}

    def cmp(self, kp):
        t0 = time.time()
        shape = kp.shape
        if kp.requires_grad:
            kp = kp.detach()
        kp = kp.cpu().flatten().numpy().astype(np.int8)
        tmp, tmp_path = mkstemp("inp_temp.bin")
        with open(tmp_path, "wb") as f:
            f.write(np.array(kp).tobytes())
        
        tmp_out, tmp_out_path = mkstemp("out_temp.bin")
        with open(tmp_path, "rb") as inp, contextlib.closing(BitOutputStream(open(tmp_out_path, "wb"))) as bitout:
            enc = ArithmeticEncoder(32, bitout)
            while True:
                sym = inp.read(1)
                if not sym:
                    break
                sym = sym[0]
                self._enc_sym(self.ppm, self.hist, sym, enc)
                self.ppm.increment_contexts(self.hist, sym)
                if self.ppm.model_order >= 1:
                    if len(self.hist) == self.ppm.model_order:
                        self.hist.pop()
                    self.hist.insert(0, sym)
            self._enc_sym(self.ppm, self.hist, 256, enc)
            enc.finish()
        
        t1 = time.time()
        res = np.reshape(self.dcmp(tmp_out_path), shape)
        t2 = time.time()
        
        for p in [tmp, tmp_out]:
            os.close(p)
        os.remove(tmp_path)
        os.remove(tmp_out_path)
        return {'bitstring_size': filesize(tmp_out_path), 'bitstring': read_bitstring(tmp_out_path), 
                'res': to_tensor(res), 'time': {'enc_time': t1-t0, 'dec_time': t2-t1}}

    def dcmp(self, in_path):
        dec_p, dec_path = mkstemp("decoding.bin")
        with open(in_path, "rb") as inp, open(dec_path, "wb") as out:
            dec = ArithmeticDecoder(32, BitInputStream(inp))
            while True:
                sym = self._dec_sym(dec, self.dec_ppm, self.dec_hist)
                if sym == 256:
                    break
                out.write(bytes((sym,)))
                self.dec_ppm.increment_contexts(self.dec_hist, sym)
                if self.dec_ppm.model_order >= 1:
                    if len(self.dec_hist) == self.dec_ppm.model_order:
                        self.dec_hist.pop()
                    self.dec_hist.insert(0, sym)
        
        with open(dec_path, 'rb') as f:
            res = np.frombuffer(f.read(), dtype=np.int8)
        os.close(dec_p)
        os.remove(dec_path)
        return res

    def _enc_sym(self, model, hist, sym, enc):
        for order in reversed(range(len(hist) + 1)):
            ctx = model.root_context
            for s in hist[:order]:
                ctx = ctx.subcontexts[s] if ctx.subcontexts else None
                if ctx is None:
                    break
            else:
                if sym != 256 and ctx.frequencies.get(sym, 0) > 0:
                    enc.write(ctx.frequencies, sym)
                    return
                enc.write(ctx.frequencies, 256)
        enc.write(model.order_minus1_freqs, sym)

    def _dec_sym(self, dec, model, hist):
        for order in reversed(range(len(hist) + 1)):
            ctx = model.root_context
            for s in hist[:order]:
                ctx = ctx.subcontexts[s] if ctx.subcontexts else None
                if ctx is None:
                    break
            else:
                sym = dec.read(ctx.frequencies)
                if sym < 256:
                    return sym
        return dec.read(model.order_minus1_freqs)
