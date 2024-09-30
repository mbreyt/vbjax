import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence, Optional, Any
from collections import namedtuple, defaultdict
from jax._src.prng import PRNGKeyArrayImpl
import jax.random as random
from vbjax.layers import MaskedMLP, OutputLayer, create_degrees, create_masks
import jax
from flax.linen.initializers import zeros
import tqdm
from .neural_mass import bold_dfun, bold_default_theta, mpr_default_theta
from flax.core.frozen_dict import freeze, unfreeze

DelayHelper = namedtuple('DelayHelper', 'Wt lags ix_lag_from max_lag n_to n_from')

class GaussianMADE(nn.Module):
    key: PRNGKeyArrayImpl
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
    key: PRNGKeyArrayImpl
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


class Buffer_step(nn.Module):
    dfun: Callable
    adhoc: Callable
    dt: float
    nh: Optional[int]
    p: Optional[Any]
    external_i: Optional[int] = False
    
    

    @nn.compact
    def __call__(self, buf, dWt, t, *args):
        t = t[0][0].astype(int) # retrieve time step
        nh = self.nh

        tmap = jax.tree_util.tree_map
        x = tmap(lambda buf: buf[nh + t], buf)


        d1 = self.dfun(buf, x, nh + t)
        xi = tmap(lambda x,d,n: x + self.dt * d + n, x, d1, dWt)
        xi = tmap(self.adhoc, xi)

        d2 = self.dfun(buf, xi, nh + t + 1)

        nx = tmap(lambda x,d1,d2,n: x + self.dt * 0.5*(d1 + d2) + n, x, d1, d2, dWt)
        nx = tmap(self.adhoc, nx)
        buf = tmap(lambda buf, nx: buf.at[nh + t + 1].set(nx), buf, nx)
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
    dfun_pars: Optional[defaultdict] = None
    dt: float = 0.1
    integrator: Optional[Callable] = Integrator
    step: Callable = Buffer_step
    adhoc: Callable = lambda x : x
    gfun: Callable = lambda x : x
    ode: bool = False

    def delay_apply(self, dh: DelayHelper, t, buf):
        return (dh.Wt * buf[t - dh.lags, dh.ix_lag_from, :]).sum(axis=1)
    
    def fwd(self, nmm, region_pars, g):
        def tvb_dfun(buf, x, t):
            coupled_x = self.delay_apply(self.tvb_p['dh'], t, buf[...,:self.nst_vars])
            coupling_term = coupled_x[:,:1] # firing rate coupling only for QIF
            # jax.debug.print("coupling {x}", x=coupling_term[0])
            return nmm(x, region_pars, g*coupling_term)
        return tvb_dfun

    def noise_fill(self, buf, nh, key):
        dWt = jax.random.normal(key, buf[nh+1:].transpose(0,2,1).shape)
        dWt = dWt.transpose(0,2,1)
        noise = self.gfun(dWt, jnp.sqrt(self.dt))
        buf = buf.at[nh+1:].set(noise)
        return buf

    def initialize_buffer(self, key):        
        dh = self.tvb_p['dh']
        nh = int(dh.max_lag)
        buf = jnp.zeros((nh + int(1/self.dt) + 1, dh.n_from, self.nst_vars))

        initial_cond = jnp.c_[
            jax.random.uniform(key=key, shape=(dh.n_from, 1), minval=0.1, maxval=2.0),
            jax.random.uniform(key=key, shape=(dh.n_from, 1), minval=-2., maxval=1.5)
            ]

        # horizon is set at the start of the buffer because rolled at the start of chunk
        buf = buf.at[int(1/self.dt):,:,:self.nst_vars].add( initial_cond )
        return buf

    def chunk(self, module, buf, key):
        nh = int(self.tvb_p['dh'].max_lag)
        buf = jnp.roll(buf, -int(1/self.dt), axis=0)
        buf = self.noise_fill(buf, nh, key)
        dWt = buf[nh+1:] # initialize carry noise filled

        # pass time count to the scanned integrator
        t_count = jnp.tile(jnp.arange(int(1/self.dt))[...,None,None], (84, 2)) # (buf_len, regions, state_vars)
        buf, rv = module(buf, dWt, t_count)
        return buf, rv

    def bold_monitor(self, module, bold_buf, rv, p=bold_default_theta):
        t_count = jnp.tile(jnp.arange(rv.shape[0])[...,None, None,None], (4, 84, 2)) # (buf_len, regions, state_vars)
        bold_buf, bold = module(bold_buf, rv, t_count)
        s, f, v, q = bold_buf
        return bold_buf, p.v0 * (p.k1 * (1. - q) + p.k2 * (1. - q / v) + p.k3 * (1. - v))
    
    @nn.compact
    def __call__(self, inputs, g, sim_len=400, seed=42):
        # i_ext = self.prepare_stimulus(x, i_ext, self.stvar)

        region_pars = inputs
        key = jax.random.PRNGKey(seed)
        buf = self.initialize_buffer(key)
        
        if self.ode:
            pars = self.param('Nodes', lambda key: unfreeze(self.dfun_pars))
            nmm = lambda x, xs, *args: self.dfun(pars, x, xs, scaling_factor=10, *args)
            tvb_dfun = self.fwd(nmm, region_pars, g)
        if self.dfun_pars:
            nmm = lambda x, xs, *args: self.dfun(self.dfun_pars, x, xs, scaling_factor=10, *args)
            tvb_dfun = self.fwd(nmm, region_pars, g)
        else:            
            nmm = lambda x, xs, *args: self.dfun(x, xs, *args)
            tvb_dfun = self.fwd(nmm, region_pars, g)


        module = self.integrator(tvb_dfun, self.step, self.adhoc, self.dt, nh=int(self.tvb_p['dh'].max_lag))
        run_chunk = nn.scan(self.chunk.__call__)
        run_sim = nn.scan(run_chunk)

        buf, rv = run_sim(module, buf, jax.random.split(key, (sim_len, 1000)))
        dummy_adhoc_bold = lambda x: x
        bold_dfun_p = lambda sfvq, x: bold_dfun(sfvq, x, bold_default_theta)
        module = self.integrator(bold_dfun_p, Heun_step, dummy_adhoc_bold, self.dt/10000, nh=int(self.tvb_p['dh'].max_lag), p=1)
        run_bold = nn.scan(self.bold_monitor.__call__)

        bold_buf = jnp.ones((4, self.tvb_p['dh'].n_from, 1))
        bold_buf = bold_buf.at[0].set(1.)


        bold_buf, bold = run_bold(module, bold_buf, rv[...,0].reshape((-1, int(20000/self.dt), self.tvb_p['dh'].n_from, 1)))

        return rv



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
    kernel_init: Callable = jax.nn.initializers.normal(1e-6)
    coupled: bool = False

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.kernel_init) for feat in self.n_hiddens]
        self.output = nn.Dense(self.out_dim, kernel_init=self.kernel_init, bias_init=self.kernel_init)
    
    @nn.compact
    def __call__(self, x, xs, *args, scaling_factor=1):
        x = jnp.c_[x, xs]
        x = jnp.c_[x, args[0]] if self.coupled else x
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        x = self.output(x)
        return x*scaling_factor


class Simple_MLP_additive_c(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    kernel_init: Callable = jax.nn.initializers.normal(1e-6)
    coupled: bool = False

    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=self.kernel_init, bias_init=self.kernel_init) for feat in self.n_hiddens]
        self.output = nn.Dense(self.out_dim, kernel_init=self.kernel_init, bias_init=self.kernel_init)
    
    @nn.compact
    def __call__(self, x, xs, *args, scaling_factor=1):
        x = jnp.c_[x, xs]
        
        for layer in self.layers:
            x = layer(x)
            x = self.act_fn(x)
        x = self.output(x)
        x += args[0] if self.coupled else x
        return x



class MontBrio(nn.Module):
    dfun_pars: Optional[defaultdict] = mpr_default_theta
    coupled: bool = False

    def setup(self):
        self.eta = self.dfun_pars['eta']
        self.Delta = self.dfun_pars['Delta']
        self.tau = self.dfun_pars['tau']
        self.I = self.dfun_pars['I']
        self.J = self.dfun_pars['J']
        self.cr = self.dfun_pars['cr']
        self.cv = self.dfun_pars['cv']
        
    
    @nn.compact
    def __call__(self, x, xs, *args):
        c = args[0] if self.coupled else jnp.zeros(x.shape)
        xs = xs
        r, V = x[:,:1], x[:,1:]
        I_c = self.cr * c[:,:1]
        r_dot =  (1 / self.tau) * (self.Delta / (jnp.pi * self.tau) + 2 * r * V)
        v_dot = (1 / self.tau) * (V ** 2 + self.eta + self.J * self.tau * r + self.I + I_c - (jnp.pi ** 2) * (r ** 2) * (self.tau ** 2))
        return jnp.c_[r_dot, v_dot]


class NeuralOdeWrapper(nn.Module):
    out_dim: int
    n_hiddens: Sequence[int]
    act_fn: Callable
    extra_p: int
    dt: Optional[float] = 1.
    step: Optional[Callable] = Heun_step
    integrator: Optional[Callable] = Integrator
    dfun: Optional[Callable] = Simple_MLP
    integrate: Optional[bool] = True
    coupled: Optional[bool] = False
    i_ext: Optional[bool] = False
    stvar: Optional[int] = 0
    adhoc: Optional[Callable] = lambda x : x
    

    @nn.compact
    def __call__(self, inputs):
        (x, i_ext) = inputs if self.coupled else (inputs, None)
        # dfun = self.dfun(self.out_dim, self.n_hiddens, self.act_fn, coupled=self.coupled)

        if not self.integrate:
            deriv = self.dfun(inputs[0], inputs[1])
            return deriv

        in_ax = (0,0,0) if self.coupled else (0,0)
        integrate = self.integrator(self.dfun.__call__, self.step, self.adhoc, self.dt, in_ax=in_ax, p=True)
        
        # xs = jnp.zeros_like(x[:,:,:int(self.extra_p)]) # initialize carry
        p = x[:,:,self.extra_p:] # initialize carry param filled
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

