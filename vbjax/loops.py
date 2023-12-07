"""
Functions for building time stepping loops.

"""

import jax
import jax.numpy as np


def heun_step(x, dfun, dt, *args, add=0, adhoc=None):
    """Use a Heun scheme to step state with a right hand sides dfun(.)
    and additional forcing term add.
    """
    adhoc = adhoc or (lambda x,*args: x)
    d1 = dfun(x, *args)
    xi = adhoc(x + dt*d1 + add, *args)
    d2 = dfun(xi, *args)
    nx = adhoc(x + dt*0.5*(d1 + d2) + add, *args)
    return nx

def make_sde(dt, dfun, gfun, adhoc=None):
    """Use a stochastic Heun scheme to integrate autonomous stochastic
    differential equations (SDEs).

    Parameters
    ==========
    dt : float
        Time step
    dfun : function
        Function of the form `dfun(x, p)` that computes drift coefficients of
        the stochastic differential equation.
    gfun : function or float
        Function of the form `gfun(x, p)` that computes diffusion coefficients
        of the stochastic differential equation. If a numerical value is
        provided, this is used as a constant diffusion coefficient for additive
        linear SDE.
    adhoc : function or None
        Function of the form `f(x, p)` that allows making adhoc corrections
        to states after a step.

    Returns
    =======
    step : function
        Function of the form `step(x, z_t, p)` that takes one step in time
        according to the Heun scheme.
    loop : function
        Function of the form `loop(x0, zs, p)` that iteratively calls `step`
        for all `z`.

    Notes
    =====

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    Note that the integrator does not sample normally distributed noise, so this
    must be provided by the user.


    >>> import vbjax as vb
    >>> _, sde = vb.make_sde(1.0, lambda x, p: -x, 0.1)
    >>> sde(1.0, vb.randn(4), None)
    Array([ 0.5093468 ,  0.30794007,  0.07600437, -0.03876263], dtype=float32)

    """

    sqrt_dt = np.sqrt(dt)

    # gfun is a numerical value or a function f(x,p) -> sig
    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    def step(x, z_t, p):
        noise = gfun(x, p) * sqrt_dt * z_t
        return heun_step(x, dfun, dt, p, add=noise, adhoc=adhoc)

    @jax.jit
    def loop(x0, zs, p):
        def op(x, z):
            x = step(x, z, p)
            return x, x
        return jax.lax.scan(op, x0, zs)[1]

    return step, loop


def make_ode(dt, dfun, adhoc=None):
    """Use a Heun scheme to integrate autonomous ordinary differential
    equations (ODEs).

    Parameters
    ==========
    dt : float
        Time step
    dfun : function
        Function of the form `dfun(x, p)` that computes derivatives of the
        ordinary differential equations.
    adhoc : function or None
        Function of the form `f(x, p)` that allows making adhoc corrections
        to states after a step.

    Returns
    =======
    step : function
        Function of the form `step(x, t, p)` that takes one step in time
        according to the Heun scheme.
    loop : function
        Function of the form `loop(x0, ts, p)` that iteratively calls `step`
        for all time steps `ts`.

    Notes
    =====

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    >>> import vbjax as vb, jax.numpy as np
    >>> _, ode = vb.make_ode(1.0, lambda x, p: -x)
    >>> ode(1.0, np.r_[:4], None)
    Array([0.5   , 0.25  , 0.125 , 0.0625], dtype=float32, weak_type=True)

    """

    def step(x, t, p):
        return heun_step(x, dfun, dt, p, adhoc=adhoc)

    @jax.jit
    def loop(x0, ts, p):
        def op(x, t):
            x = step(x, t, p)
            return x, x
        return jax.lax.scan(op, x0, ts)[1]

    return step, loop


def make_dde(dt, nh, dfun, unroll=10, adhoc=None):
    "Invokes make_sdde w/ gfun 0."
    return make_sdde(dt, nh, dfun, 0, unroll, adhoc=adhoc)


def make_sdde(dt, nh, dfun, gfun, unroll=1, zero_delays=False, adhoc=None):
    """Use a stochastic Heun scheme to integrate autonomous
    stochastic delay differential equations (SDEs).

    Parameters
    ==========
    dt : float
        Time step
    nh : int
        Maximum delay in time steps.
    dfun : function
        Function of the form `dfun(xt, x, t, p)` that computes drift coefficients of
        the stochastic differential equation.
    gfun : function or float
        Function of the form `gfun(x, p)` that computes diffusion coefficients
        of the stochastic differential equation. If a numerical value is
        provided, this is used as a constant diffusion coefficient for additive
        linear SDE.
    adhoc : function or None
        Function of the form `f(x,p)` that allows making adhoc corrections after
        each step.

    Returns
    =======
    step : function
        Function of the form `step((x_t,t), z_t, p)` that takes one step in time
        according to the Heun scheme.
    loop : function
        Function of the form `loop((xs, t), p)` that iteratively calls `step`
        for each `xs[nh:]` and starting time index `t`.

    Notes
    =====

    - A Jax compatible parameter set `p` is provided, either an array
      or some pytree compatible structure.
    - The integrator does not sample normally distributed noise, so this
      must be provided by the user.
    - The history buffer passed to the user functions, on the corrector
      stage of the Heun method, does not contain the predictor stage, for
      performance reasons, unless `zero_delays` is set to `True`.
      A good compromise can be to set all zero delays to dt.

    >>> import vbjax as vb, jax.numpy as np
    >>> _, sdde = vb.make_sdde(1.0, 2, lambda xt, x, t, p: -xt[t-2], 0.0)
    >>> x,t = sdde(np.ones(6)+10, None)
    >>> x
    Array([ 11.,  11.,  11.,   0., -11., -22.], dtype=float32)

    """

    heun = True
    sqrt_dt = np.sqrt(dt)

    # TODO nh == 0: return make_sde(dt, dfun, gfun) etc

    # gfun is a numerical value or a function f(x,p) -> sig
    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    if adhoc is None:
        adhoc = lambda x,p : x

    def step(buf_t, z_t, p):
        buf, t = buf_t
        x = buf[nh + t]
        noise = gfun(x, p) * sqrt_dt * z_t
        d1 = dfun(buf, x, nh + t, p)
        xi = adhoc(x + dt*d1 + noise, p)
        if heun:
            if zero_delays:
                # severe performance hit (5x+)
                buf = buf.at[nh + t + 1].set(xi)
            d2 = dfun(buf, xi, nh + t + 1, p)
            nx = adhoc(x + dt*0.5*(d1 + d2) + noise, p)
        else:
            nx = xi
        buf = buf.at[nh + t + 1].set(nx)
        return (buf, t+1), nx

    @jax.jit
    def loop(buf, p, t=0):
        "xt is the buffer, zt is (ts, zs), p is parameters."
        op = lambda xt, tz: step(xt, tz, p)
        dWt = buf[nh:]
        (buf, _), nxs = jax.lax.scan(op, (buf, t), dWt, unroll=unroll)
        return buf, nxs

    return step, loop

