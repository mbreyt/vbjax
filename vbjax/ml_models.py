import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence, Optional, Any
from collections import namedtuple, defaultdict
# from jax._src.prng import PRNGKeyArrayImpl
import jax.random as random
from vbjax.layers import MaskedMLP, OutputLayer, create_degrees, create_masks, OutputLayerAdditive
import jax
from flax.linen.initializers import zeros
import tqdm
from .neural_mass import bold_dfun, bold_default_theta
from flax.core.frozen_dict import freeze, unfreeze


DelayHelper = namedtuple('DelayHelper', 'Wt lags ix_lag_from max_lag n_to n_from')

class GaussianMADE(nn.Module):
    # key: PRNGKeyArrayImpl
    in_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    input_order: str = 'sequential'
    mode: str = 'sequential'

    def setup(self):
        self.degrees = create_degrees(self.key, self.in_dim, self.n_hiddens, input_order=self.input_order, mode=self.mode)
        self.masks, self.out_mask = create_masks(self.degrees)
        self.mlp = MaskedMLP(self.n_hiddens, self.act_fn, self.masks)
        self.output_layer = OutputLayer(self.in_dim, self.out_mask)

    
    def __call__(self, inputs):
        h = self.mlp(inputs)
        m, logp = self.output_layer(h)
        return m, logp


    def gen(self, key, shape, u=None):
        x = jnp.zeros(shape)
        u = random.normal(key, shape) if u is None else u

        for i in range(1, shape[1] + 1):
            h = self.mlp(x)
            m, logp = self.output_layer(h) 
            idx = jnp.argwhere(self.degrees[0] == i)[0, 0]
            x = x.at[:, idx].set(m[:, idx] + jnp.exp(jnp.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx])
        return x


class MAF(nn.Module):
    # key: PRNGKeyArrayImpl
    in_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    n_mades: int
    input_order: Optional[Sequence] = None
    mode: str = 'sequential'

    def setup(self, input_order: Optional[Sequence] = None):
        input_order = jnp.arange(1, self.in_dim+1) if input_order == None else input_order
        self.mades = [GaussianMADE(random.split(self.key), self.in_dim, self.n_hiddens, self.act_fn, input_order=input_order[::((-1)**(i%2))], mode=self.mode) for i in range(self.n_mades)]

    def __call__(self, inputs):
        u = inputs
        logdet_dudx = 0
        for made in self.mades:
            ms, logp = made(u)
            u = jnp.exp(0.5 * logp) * (u - ms)
            # jax.debug.print('logp {x}', x=jnp.mean(jnp.sum(logp, axis=1)))
            logdet_dudx += 0.5 * jnp.sum(logp, axis=1)
        return u, logdet_dudx

    def gen(self, key, shape, u=None):
        x = random.normal(key, shape) if u is None else u

        for made in self.mades[::-1]:
            x = made.gen(key, shape, x)
        return x


class Heun_step(nn.Module):
    dfun: Callable
    adhoc: Callable
    dt: float
    nh: Optional[int]
    p: Optional[Any]
    stvar: Optional[int] = 0
    external_i: Optional[int] = False
    

    @nn.compact
    def __call__(self, x, xs, t, *args):
        tmap = jax.tree_util.tree_map
        d1 = self.dfun(x, xs, *args) if self.p else self.dfun(x, *args)
        xi = tmap(lambda x,d: x + self.dt*d, x, d1)
        xi = tmap(self.adhoc, xi)

        d2 = self.dfun(xi, xs, *args) if self.p else self.dfun(xi, *args)
        nx = tmap(lambda x, d1,d2: x + self.dt*0.5*(d1 + d2), x, d1, d2)
        nx = tmap(self.adhoc, nx)
        return nx, x


class Euler_step(nn.Module):
    dfun: Callable
    adhoc: Callable
    dt: float
    nh: Optional[int]
    p: Optional[Any]
    stvar: Optional[int] = 0
    external_i: Optional[int] = False
    

    @nn.compact
    def __call__(self, x, xs, t, *args):
        tmap = jax.tree_util.tree_map
        d1 = self.dfun(x, xs, *args) if self.p else self.dfun(x, *args)
        nx = tmap(lambda x,d: x + self.dt*d, x, d1)
        nx = tmap(self.adhoc, nx)
        return nx, x


class Heun_step_stim(nn.Module):
    dfun: Callable
    adhoc: Callable
    dt: float
    nh: Optional[int]
    p: Optional[Any]
    stvar: Optional[int] = 0
    external_i: Optional[int] = False
    

    @nn.compact
    def __call__(self, x, xs, t, *args):
        tmap = jax.tree_util.tree_map
        d1 = self.dfun(x, xs, *args) if self.p else self.dfun(x, *args)
        xi = tmap(lambda x,d: x + self.dt*d, x, d1)
        xi = tmap(self.adhoc, xi)

        d2 = self.dfun(xi, xs, *args) if self.p else self.dfun(xi, *args)
        nx = tmap(lambda x, d1,d2: x + self.dt*0.5*(d1 + d2), x, d1, d2)
        nx = tmap(self.adhoc, nx)
        return nx, x



class Buffer_step(nn.Module):
    dfun: Callable
    adhoc: Callable
    dt: float
    nh: Optional[int]
    p: Optional[Any]
    external_i: Optional[int] = False
    
    

    @nn.compact
    def __call__(self, buf, dWt, t, *args):
        # jax.debug.print('t buffer step {x}', x=t)
        t_step = t.at[0,0].get().astype(int) # retrieve time step
        stim = t.at[1,:].get()
        nh = self.nh
        tmap = jax.tree_util.tree_map
        x = tmap(lambda buf: buf[nh + t_step], buf)
        # jax.debug.print('buffer {x}', x=t)
        # jax.debug.print('x {x}', x=x.shape)
        d1 = self.dfun(buf, x, nh + t_step,  t)
        xi = tmap(lambda x,d,n: x + self.dt * d + n, x, d1, dWt)
        xi = tmap(self.adhoc, xi)

        d2 = self.dfun(buf, xi, nh + t_step + 1,  t)

        nx = tmap(lambda x,d1,d2,n: x + self.dt * 0.5*(d1 + d2) + n, x, d1, d2, dWt)
        nx = tmap(self.adhoc, nx)
        buf = tmap(lambda buf, nx: buf.at[nh + t_step + 1].set(nx), buf, nx)
        return buf, nx


class Buffer_step_euler(nn.Module):
    dfun: Callable
    adhoc: Callable
    dt: float
    nh: Optional[int]
    p: Optional[Any]
    external_i: Optional[int] = False
    
    @nn.compact
    def __call__(self, buf, dWt, t, *args):
        t_step = t.at[0,0].get().astype(int) # retrieve time step
        nh = self.nh
        tmap = jax.tree_util.tree_map
        x = tmap(lambda buf: buf[nh + t_step], buf)
        d1 = self.dfun(buf, x, nh + t_step,  t)
        xi = tmap(lambda x,d,n: x + self.dt * d + n, x, d1, dWt)
        nx = tmap(self.adhoc, xi)

        # d2 = self.dfun(buf, xi, nh + t_step + 1,  t)
        # nx = tmap(lambda x,d1,d2,n: x + self.dt * 0.5*(d1 + d2) + n, x, d1, d2, dWt)
        nx = tmap(self.adhoc, nx)
        buf = tmap(lambda buf, nx: buf.at[nh + t_step + 1].set(nx), buf, nx)
        return buf, nx



class Integrator(nn.Module):
    dfun: Callable
    step: Callable
    adhoc: Callable
    dt: float = 1.0
    stvar: Optional[int] = 0
    nh: Optional[int] = None
    p: Optional[Any] = True
    in_ax: Optional[tuple] = (0,0)

    @nn.compact
    def __call__(self, c, xs, t_count, *args):
        STEP = nn.scan(self.step,
                        # variable_broadcast=["params", "noise"],
                        # split_rngs={"params": False, "noise": True},
                        variable_broadcast=["params"],
                        split_rngs={"params": False},                        
                        in_axes=self.in_ax,
                        out_axes=0
                        )
        return STEP(self.dfun, self.adhoc, self.dt, self.nh, self.p)(c, xs, t_count, *args)



class TVB(nn.Module):
    tvb_p: namedtuple
    dfun: Callable
    nst_vars: int
    n_pars: int
    # dfun_pars: Optional[defaultdict] = jnp.array([])
    dfun_pars: Optional[Any] = None
    dt: float = 0.1
    integrator: Optional[Callable] = Integrator
    step: Callable = Buffer_step
    adhoc: Callable = lambda x : x
    gfun: Callable = lambda x : x
    # stimulus: Optional[Sequence] = jnp.array([])
    stimulus: Optional[Sequence] = None
    node_stim = 0
    training: bool = False
    bold_buf_size: int = 2000
    bold_dt: float = 0.01
    chunksize: int = 1000
    tavg_period: float = 1.
    # initial_cond: Optional[jnp.array] = jnp.array([])
    initial_cond: Optional[jnp.array] = None

    def delay_apply(self, dh: DelayHelper, t, buf):
        return (dh.Wt * buf[t - dh.lags, dh.ix_lag_from, :]).sum(axis=1)
    
    def fwd(self, nmm, region_pars, g):
        def tvb_dfun(buf, x, t, stim):
            coupled_x = self.delay_apply(self.tvb_p['dh'], t, buf[...,:self.nst_vars])
            coupling_term = coupled_x[:,:1] # firing rate coupling only for QIF
            return nmm(x, region_pars, g*coupling_term+stim[:,1:])
        return tvb_dfun

    def noise_fill(self, buf, nh, key):
        dWt = jax.random.normal(key, buf[nh+1:].transpose(0,2,1).shape)
        dWt = dWt.transpose(0,2,1)
        noise = self.gfun(dWt, jnp.sqrt(self.dt))
        buf = buf.at[nh+1:].set(noise)
        return buf

    def initialize_buffer(self, key, fixed_initial_cond):        
        dh = self.tvb_p['dh']
        nh = int(dh.max_lag)
        buf = jnp.zeros((nh + int(1/self.dt) + 1, dh.n_from, self.nst_vars))
        initial_cond = jnp.c_[
            jax.random.uniform(key=key, shape=(dh.n_from, 1), minval=0.1, maxval=2.0),
            jax.random.uniform(key=key, shape=(dh.n_from, 1), minval=-2., maxval=1.5)
            ]
        initial_cond = self.initial_cond if fixed_initial_cond else initial_cond

        # horizon is set at the start of the buffer because rolled at the start of chunk
        buf = buf.at[int(1/self.dt):,:,:self.nst_vars].add( initial_cond )
        return buf

    def chunk(self, module, buf, stimulus, key):
        nh = int(self.tvb_p['dh'].max_lag) 
        buf = jnp.roll(buf, -int(1/self.dt), axis=0)
        buf = self.noise_fill(buf, nh, key)
        dWt = buf[nh+1:] # initialize carry noise filled
        # jax.debug.print('stim {x}', x=stimulus)
        # pass time count to the scanned integrator
        t_count = jnp.tile(jnp.arange(int(1/self.dt))[...,None,None], (self.tvb_p['dh'].n_from, 1)) # (buf_len, regions, state_vars)
        stim = jnp.zeros(t_count.shape)
        # stimulus = jnp.repeat(stimulus, int(1/self.dt))[...,None]
        stim = stim.at[:,:,:].set(jnp.tile(stimulus[...,None], self.tvb_p['dh'].n_from)[...,None]) if self.training else stim.at[:,self.node_stim,:].set(stimulus[...,None])
        stim_t_count = jnp.c_[t_count, stim]
        buf, rv = module(buf, dWt, stim_t_count)
        # jax.debug.print('rv shape {x}', x=rv.shape)
        # jax.debug.print('buf shape {x}', x=buf.shape)
        return buf, jnp.mean(rv.reshape(-1, int(1/self.tavg_period), self.tvb_p['dh'].n_from, 2), axis=1)

    def bold_monitor(self, module, bold_buf, rv, p=bold_default_theta):
        t_count = jnp.tile(jnp.arange(rv.shape[0])[...,None, None,None], (4, self.tvb_p['dh'].n_from, 2)) # (buf_len, regions, state_vars)
        bold_buf, bold = module(bold_buf, rv, t_count)
        s, f, v, q = bold_buf
        return bold_buf, p.v0 * (p.k1 * (1. - q) + p.k2 * (1. - q / v) + p.k3 * (1. - v))
    


    @nn.compact
    def __call__(self, region_pars, g=0, sim_len=0, seed=42, initial_cond=False, mlp=True, stimulus_yn=True):
        # if inputs==None:
        #     inputs = jnp.ones((1, self.nst_vars))
        key = jax.random.PRNGKey(seed)
        # buf = self.initialize_buffer(key, initial_cond)
        
        if mlp:
            nmm = lambda x, xs, *args: self.dfun(self.dfun_pars, x, xs, *args)
            tvb_dfun = self.fwd(nmm, region_pars, g)
        else:            
            nmm = lambda x, xs, *args: self.dfun(x, xs, *args)
            tvb_dfun = self.fwd(nmm, region_pars, g)

        # nmm = lambda x, xs, *args: self.dfun(self.dfun_pars, x, xs, *args) if mlp else self.dfun.__call__
        # tvb_dfun = self.fwd(nmm, region_pars, g)

        module = self.integrator(tvb_dfun, self.step, self.adhoc, self.dt, nh=int(self.tvb_p['dh'].max_lag))
        run_chunk = nn.scan(self.chunk.__call__)
        run_sim = nn.scan(run_chunk)
        
        buf = self.initialize_buffer(key, initial_cond)
        
        stimulus = self.stimulus.reshape((sim_len, int(self.chunksize*self.dt), -1)) if stimulus_yn else jnp.zeros((sim_len, int(self.chunksize*self.dt), 1))
        # stimulus = jnp.zeros((sim_len, int(self.chunksize*self.dt), 1))
        # jax.debug.print('chuksize {x}', x=self.chunksize)
        # jax.debug.print('dt {x}', x=self.dt)
        # jax.debug.print('stim shape {x}', x=stimulus.shape)
        # jax.debug.print('buf shape {x}', x=buf.shape)
        # jax.debug.print('keys shape {x}', x=jax.random.split(key, (sim_len, int(self.chunksize*self.dt))).shape)
        buf, rv = run_sim(module, buf, stimulus, jax.random.split(key, (sim_len, int(self.chunksize*self.dt))))

        dummy_adhoc_bold = lambda x: x
        bold_dfun_p = lambda sfvq, x: bold_dfun(sfvq, x, bold_default_theta)
        module = self.integrator(bold_dfun_p, Heun_step, dummy_adhoc_bold, self.bold_dt, nh=int(self.tvb_p['dh'].max_lag), p=1)
        run_bold = nn.scan(self.bold_monitor.__call__)

        bold_buf = jnp.ones((4, self.tvb_p['dh'].n_from, 1))
        bold_buf = bold_buf.at[0].set(1.)

        bold_buf, bold = run_bold(module, bold_buf, rv[...,0].reshape((-1, int(self.bold_buf_size), self.tvb_p['dh'].n_from, 1)))
        return rv.reshape(-1, self.tvb_p['dh'].n_from, self.nst_vars), bold



class TVB_ODE(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable

    def setup(self):
        self.Nodes = nn.vmap(
                    Simple_MLP,
                    in_axes=0, out_axes=0,
                    variable_axes={'params': 0},
                    split_rngs={'params': True},
                    methods=["__call__"])(out_dim=self.out_dim, n_hiddens=self.n_hiddens, act_fn=self.act_fn, coupled=True)

    def __call__(self, x, xs, *args):
        y = self.Nodes(x, xs, *args)
        return y
        


class Simple_MLP(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    kernel_init: Callable = jax.nn.initializers.normal(1e-3)
    # kernel_init: Callable = None
    coupled: bool = False
    n_pars: int = 0
    scaling_factor: float = .01

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros) for feat in self.n_hiddens]
        self.output = nn.Dense(self.out_dim, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)
    
    @nn.compact
    def __call__(self, x, xs, *args):
        # c = args[0]
        # jax.debug.print('x[0] {x} xs[0] {y} args[0] {z}', x=x.shape, y=xs.shape, z=c.shape)
        # jax.debug.print('x[0] {x} xs[0] {y}', x=x.shape, y=xs.shape)
        
        x = jnp.c_[x, xs]
        x = jnp.c_[x, args[0]] if self.coupled else x
        # jax.debug.print('x {x}', x=x)
        # jax.debug.print('x[0] {x}', x=x[:2])
        # jax.debug.print('x[0] {x}', x=x[0])
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        x = self.output(x)
        
        return x*self.scaling_factor


class Simple_MLP_additive_c(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    # kernel_init: Callable = None
    kernel_init: Callable = jax.nn.initializers.normal(1e-3)
    coupled: bool = False
    scaling_factor: float = 1.

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros) for feat in self.n_hiddens]
        self.output = nn.Dense(self.out_dim, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros) # use_bias=False
    
    @nn.compact
    def __call__(self, x, xs, *args):
        # c = args[0]
        # jax.debug.print('x[0] {x} xs[0] {y} args[0] {z}', x=x[0], y=xs[0], z=c[0])
        x = jnp.c_[x, xs]
        # jax.debug.print('x[0] {x}', x=x[:2])
        for layer in self.layers:
            x = layer(x)    
            x = self.act_fn(x)
        x = self.output(x)
    
        x = x*self.scaling_factor
        
        # jax.debug.print('x[0] {x} xs[0] {y} args[0] {z}', x=x[0], y=xs[0], z=c[0])
        # jax.debug.print('x.shape {x} xs.shape {y} args.shape {z}', x=x.shape, y=xs.shape, z=c.shape)
        # jax.debug.print('before x[0] {x}', x=x[0])
        x += jnp.c_[jnp.zeros(args[0].shape), args[0]] if self.coupled else x
        # jax.debug.print('after x[0] {x}', x=x[0])
        return x



class Additive_c(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    kernel_init: Callable = None
    coupled: bool = False
    scaling_factor: float = 1.

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros) for feat in self.n_hiddens]
        self.output = nn.OutputLayerAdditive(self.out_dim, kernel_init=self.kernel_init)#, bias_init=nn.initializers.zeros)
    
    @nn.compact
    def __call__(self, x, xs, *args):
        # c = args[0]
        # jax.debug.print('x[0] {x} xs[0] {y} args[0] {z}', x=x[0], y=xs[0], z=c[0])
        x = jnp.c_[x, xs]
        for layer in self.layers:
            x = layer(x)    
            x = self.act_fn(x)
        x = self.output(x)
    
        x = x*self.scaling_factor
        # jax.debug.print('x[0] {x}', x=x[0])
        # jax.debug.print('c[0] {x}', x=c[0])
        # jax.debug.print('x[0] {x} xs[0] {y} args[0] {z}', x=x[0], y=xs[0], z=c[0])
        x += jnp.c_[jnp.zeros(args[0].shape), args[0]] if self.coupled else x
        # jax.debug.print('after x[0] {x}', x=x[0])
        return x



class MontBrio(nn.Module):
    # dfun_pars: None #Optional[defaultdict] = mpr_default_theta
    dfun_pars: Optional[defaultdict] = None
    coupled: bool = False
    scaling_factor: float = 1.
    Delta: Optional[float] = 1.0
    tau: Optional[float] = 1.0
    I: Optional[float] = 0
    J: Optional[float] = 15.0
    cr: Optional[float] = 1.0
    cv: Optional[float] = 0.0
    
    @nn.compact
    def __call__(self, x, xs, *args):
        # xs contains regions parameters not implemented yet
        c = args[0] if self.coupled else jnp.zeros(x.shape)
        eta, J = xs[:,:1], xs[:,1:]
        r, V = x[:,:1], x[:,1:]
        I_c = self.cr * c[:,:1]
        r_dot =  (1 / self.tau) * (self.Delta / (jnp.pi * self.tau) + 2 * r * V)
        v_dot = (1 / self.tau) * (V ** 2 + eta + J * self.tau * r + self.I + I_c - (jnp.pi ** 2) * (r ** 2) * (self.tau ** 2))
        return jnp.c_[r_dot, v_dot]*self.scaling_factor


class NeuralOdeWrapper(nn.Module):
    out_dim: int
    extra_p: int
    dt: Optional[float] = 1.
    step: Optional[Callable] = Heun_step
    integrator: Optional[Callable] = Integrator
    dfun: Optional[Callable] = None
    integrate: Optional[bool] = True
    coupled: Optional[bool] = False
    stvar: Optional[int] = 0
    adhoc: Optional[Callable] = lambda x : x
    

    @nn.compact
    def __call__(self, inputs, integrate=True, additive=False):
        # jax.debug.print('inputs {x}', x=inputs.shape)
        
        # dfun = self.dfun(self.out_dim, self.n_hiddens, self.act_fn, coupled=self.coupled)
        if not integrate:
            # x = jnp.c_[inputs[0], inputs[1]]
            if additive:
                # jax.debug.print('deriv {x}', x=(inputs[0][:2], inputs[1][:2], inputs[2][:2]))
                deriv = self.dfun(inputs[0], inputs[1], inputs[2])
            else:
                # jax.debug.print('deriv {x}', x=(inputs[0][:2], inputs[1][:2]))
                deriv = self.dfun(inputs[0], inputs[1])
            # jax.debug.print('deriv {x}', x=deriv[0])
            return deriv
        
        (x, i_ext) = inputs if self.coupled else (inputs, None)
        in_ax = (0,0,0) if self.coupled else (0,0)
        integrate = self.integrator(self.dfun.__call__, self.step, self.adhoc, self.dt, in_ax=in_ax, p=True)
        
        # xs = jnp.zeros_like(x[:,:,:int(self.extra_p)]) # initialize carry
        p = x[:,:,-self.extra_p:] # initialize carry param filled
        # jax.debug.print('p {x}', x=p.shape)
        # i_ext = self.prepare_stimulus(x, i_ext, self.stvar)
        t_count = jnp.tile(jnp.arange(x.shape[0])[...,None,None], (x.shape[1], x.shape[2])) # (length, train_samples, state_vars)
        
        x = x[0,:,:self.out_dim]
        return integrate(x, p, t_count, i_ext)[1]


class Encoder(nn.Module):
    in_dim: int
    latent_dim: int
    act_fn: Callable
    n_hiddens: Sequence[int] = None

    def setup(self, n_hiddens: Optional[Sequence] = None):
        n_hiddens = n_hiddens[::-1] if n_hiddens else [self.in_dim, 4*self.latent_dim, 2*self.latent_dim, self.latent_dim][::-1]
        self.layers = [nn.Dense(feat) for feat in n_hiddens]

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        return x


class Decoder(nn.Module):
    in_dim: int
    latent_dim: int
    act_fn: Callable
    n_hiddens: Sequence[int] = None

    def setup(self, n_hiddens: Optional[Sequence] = None):
        n_hiddens = n_hiddens[::-1] if n_hiddens else [self.in_dim, 4*self.latent_dim, 2*self.latent_dim, self.latent_dim][::-1]
        self.layers = [nn.Dense(feat) for feat in n_hiddens]

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fn(x)
        x = self.layers[-1](x)
        return x


class Autoencoder(nn.Module):
    latent_dim: int
    encoder_act_fn: Callable
    decoder_act_fn: Callable
    ode_act_fn: Callable
    ode: bool = False
    n_hiddens: Sequence[int] = None
    kernel_init: Callable = jax.nn.initializers.normal(10e-3)
    step: Optional[Callable] = Heun_step
    integrator: Optional[Callable] = Integrator
    network: Optional[Callable] = Simple_MLP
    i_ext: Optional[bool] = True
    ode_n_hiddens: Optional[Sequence] = None

    def integrate(self, encoded, L):
        xs =  jnp.ones((encoded.shape[0], encoded.shape[1], L)) # initialize carry
        dfun = self.network(encoded.shape[1], self.ode_n_hiddens, self.ode_act_fn)
        integrator = self.integrator(dfun, self.step)
        return integrator(encoded, xs)[1]

    @nn.compact
    def __call__(self, inputs):
        L = inputs.shape[-1]

        encoder = Encoder(inputs.shape[1], self.latent_dim, self.encoder_act_fn, self.n_hiddens)
        encoded = encoder(inputs[:,:,0]) if self.ode else encoder(inputs) # (N, )

        decoder = Decoder(inputs.shape[1], self.latent_dim, self.decoder_act_fn)
        y = decoder(encoded)
        if self.ode:
            y = self.integrate(encoded, L)

        return y



def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    z = mean + std*eps
    return z, mean, logvar


class DUMMY(nn.Module):
    act_fn: Callable
    kernel_init: Callable = jax.nn.initializers.normal(1e-1)
    
    @nn.compact
    def __call__(self, x, z_ng):
        x = nn.Dense(2, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        x = self.act_fn(x)
        x = nn.Dense(2, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        x = self.act_fn(x)
        z, mean, logvar = reparameterize(z_ng, x[...,:1], x[...,1:])
        return z, mean, logvar
    

class NeuralOdeWrapper2(nn.Module):
    out_dim: int
    extra_p: int
    dt: Optional[float] = 1.
    step: Optional[Callable] = Heun_step
    integrator: Optional[Callable] = Integrator
    dfun: Optional[Callable] = None
    integrate: Optional[bool] = True
    coupled: Optional[bool] = False
    stvar: Optional[int] = 0
    adhoc: Optional[Callable] = lambda x : x
    dummy : Optional[Callable] = None
    

    @nn.compact
    def __call__(self, inputs, z_rng, integrate=True, additive=False):
        if not integrate:
            if additive:
                deriv = self.dfun(inputs[0], inputs[1], inputs[2])
            else:
                z = self.dummy(inputs[0][...,:2][None,...], z_rng)
                deriv = self.dfun(inputs[0], z[0])
            return deriv
        
        (x, i_ext) = inputs if self.coupled else (inputs, None)
        in_ax = (0,0,0) if self.coupled else (0,0)
        integrate = self.integrator(self.dfun.__call__, self.step, self.adhoc, self.dt, in_ax=in_ax, p=True)
        
        z = self.dummy(x[:,:,:2], z_rng)
        
        p = x[:,:,-self.extra_p:] # initialize carry param filled

        t_count = jnp.tile(jnp.arange(x.shape[0])[...,None,None], (x.shape[1], x.shape[2])) # (length, train_samples, state_vars)
        
        x = x[0,:,:self.out_dim]
        return integrate(x, p, t_count, z)[1]#, recon_m, recon_logvar, mean, logvar
    

def sample_i_ext(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, mean.shape)
  return mean + eps


class DUMMY2(nn.Module):
    act_fn: Callable
    kernel_init: Callable = jax.nn.initializers.normal(1e-3)
    
    @nn.compact
    def __call__(self, x, z_ng):
        x = nn.Dense(2, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        x = self.act_fn(x)
        x = nn.Dense(2, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        y = sample_i_ext(z_ng, x[0,:,:1], x[0,:,1:])[None,...]
        x = jnp.tile(y, (x.shape[0], 1, 1))
        return x
    


class Encoder(nn.Module):
    act_fn: Callable
    n_hiddens: Sequence[int]
    kernel_init: Callable = jax.nn.initializers.normal(10e-3)

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init) for feat in self.n_hiddens]
        self.mean = nn.Dense(self.n_hiddens[-1], name='fc2_mean', kernel_init=self.kernel_init)
        self.logvar = nn.Dense(self.n_hiddens[-1], name='fc2_logvar', kernel_init=self.kernel_init)

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fn(x)
        mean_x = self.mean(x)
        logvar_x = self.logvar(x)
        return mean_x, logvar_x



class Decoder(nn.Module):
    act_fn: Callable
    n_hiddens: Sequence[int]
    kernel_init: Callable = jax.nn.initializers.normal(10e-3)

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init) for feat in self.n_hiddens[::-1]]
        self.mean = nn.Dense(self.n_hiddens[::-1][-1], name='decod_mean', kernel_init=self.kernel_init)
        self.logvar = nn.Dense(self.n_hiddens[::-1][-1], name='decod_logvar', kernel_init=self.kernel_init)

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fn(x)
        x = self.layers[-1](x)
        return self.mean(x), self.logvar(x)


def reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


class VAE(nn.Module):
    act_fn: Callable
    n_hiddens: Sequence[int]

    def setup(self):
        self.encoder = Encoder(self.act_fn, self.n_hiddens[1:]) # remove input dim
        self.decoder = Decoder(self.act_fn, self.n_hiddens)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_m, recon_logvar = self.decoder(z)
        # recon_x = self.decoder(mean)
        return recon_m, recon_logvar, z, mean, logvar

    def generate(self, z, z_rng):
        return jax.random.poisson(z_rng, self.decoder(z))
