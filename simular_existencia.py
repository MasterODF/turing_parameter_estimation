"""
Simulations for question 4 of the paper: for which (a, Dv) does a Turing pattern form?

The data are generated with a BayesFlow simulator made of two parts:
    prior    a ~ U(A_MIN, A_MAX), Dv ~ U(0, DV_MAX), and a seed for the initial noise
    modelo   the reaction-diffusion model of simular_constante.py, on a periodic L x L grid (dx = 1), integrated with
             explicit Euler from u = v = 2 plus Gaussian noise:
                 du/dt = a (u + v - 4) - (u - 2)^3 / 2 + lap(u)
                 dv/dt = 6 - 2u - v + Dv lap(v)
             it returns the spatial standard deviation sigma of u, which decays towards 0 when the homogeneous state
             is stable and saturates at O(1) when a pattern forms
Dv is drawn on both sides of the Turing threshold Dv_min(a) = (3 + 2 sqrt(2)) / a. The model is the expensive part,
so it is batched: BayesFlow hands it all the (a, Dv) of a set at once and it runs them in parallel processes.

Outputs of the simulator, stored per set:
    a, Dv, semilla       parameters and seed of the initial noise
    sigma                standard deviation of u at the final time (exactly 0 if u is uniform to machine precision)
    log_sigma            log10(max(sigma, SIGMA_MINIMO)), the quantity learned by inferir_existencia.py
    patron               sigma > UMBRAL (absolute threshold)
    crece                sigma > sigma(0) (the initial perturbation has grown)
    sigma_t              sigma every INTERVALO time units, to revisit the criterion without simulating again
Close to the threshold (and close to a = 1, where the homogeneous state approaches a Hopf bifurcation) perturbations
decay or grow very slowly, so an absolute threshold on sigma(T) can misclassify them; `crece` is more robust.

Writes to the output folder:
    entrenar.npz, validar.npz, test.npz   the outputs above, with shape (N, 1) except sigma_t (N, M), and tiempos (M,)
    parametros.json                       every setting, so the batch can be reproduced
    patron_<set>.png                      every simulation in the (a, Dv) plane: green with a pattern
                                          (sigma(T) > sigma(0)), black without

The defaults are the settings used in the paper:
    python simular_existencia.py
"""

import os
import sys
import json
import time
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# One thread per process: the parallelism comes from the process pool
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import jit

CONJUNTOS = ('entrenar', 'validar', 'test')
# sigma below this value is noise around machine precision (u is about 2); it is also the floor of log_sigma
SIGMA_MINIMO = 1e-16


''' MODEL '''

# 5-point Laplacian with periodic boundaries
@jit(nopython=True, cache=True)
def laplaciano(matriz):
    n, m = matriz.shape
    lap = np.empty_like(matriz)
    for i in range(n):
        for j in range(m):
            lap[i, j] = (matriz[(i+1) % n, j] + matriz[(i-1) % n, j]
                         + matriz[i, (j+1) % m] + matriz[i, (j-1) % m]
                         - 4 * matriz[i, j])
    return lap


# Explicit Euler from the u = v = 2 equilibrium plus noise; returns the standard deviation of u at the iterations in
# iteraciones_guardar (sorted, starting at 0)
@jit(nopython=True, cache=True)
def simular(a, Dv, L, num_iter, DELTA_T, ruido, semilla, iteraciones_guardar):
    np.random.seed(semilla)
    u = 2 + ruido*np.random.standard_normal((L, L))
    v = 2 + ruido*np.random.standard_normal((L, L))
    sigma = np.empty(len(iteraciones_guardar))
    siguiente = 0
    for iteracion in range(num_iter + 1):
        while siguiente < len(iteraciones_guardar) and iteraciones_guardar[siguiente] == iteracion:
            sigma[siguiente] = u.std()
            siguiente += 1
        if iteracion == num_iter:
            break
        u_nuevo = u + DELTA_T*(a*(u + v - 4) - 0.5*(u - 2)**3 + laplaciano(u))
        v_nuevo = v + DELTA_T*(6 - (2*u + v) + Dv*laplaciano(v))
        u, v = u_nuevo, v_nuevo
    return sigma


# Turing instability threshold: minimum Dv for a given a (Jacobian at u = v = 2 is [[a, a], [-2, -1]])
def D_minimo(a):
    return (3 + 2*np.sqrt(2)) / a


# One simulation, run in a worker process. Explicit Euler is stable if Dv*DELTA_T/dx**2 <= 0.25
def ejecutar(a, Dv, semilla, cfg):
    DELTA_T = min(cfg['dt_max'], cfg['factor_estabilidad'] / max(Dv, 1e-12))
    num_iter = int(cfg['t_final'] / DELTA_T)
    tiempos = np.asarray(cfg['tiempos'])
    iteraciones_guardar = np.minimum(np.round(tiempos / DELTA_T).astype(np.int64), num_iter)
    return simular(a, Dv, cfg['L'], num_iter, DELTA_T, cfg['ruido'], semilla, iteraciones_guardar)


''' BAYESFLOW SIMULATOR '''

# Prior: batched, draws all the parameters of a set from its own random stream
def hacer_prior(rng, args):
    def prior(batch_shape):
        num = int(np.prod(batch_shape))
        a = rng.uniform(args.a_min, args.a_max, size=num)
        return {'a': a,
                'Dv': rng.uniform(0, args.dv_max, size=num),
                'semilla': rng.integers(2**31, size=num)}
    return prior


# Model: batched, runs every (a, Dv) of the batch in the process pool and returns sigma and the derived quantities
def hacer_modelo(executor, cfg, args, nombre):
    def modelo(batch_shape, a, Dv, semilla):
        a, Dv, semilla = np.ravel(a), np.ravel(Dv), np.ravel(semilla)
        sigma_t = np.empty((len(a), len(cfg['tiempos'])))
        futuros = {executor.submit(ejecutar, float(a[i]), float(Dv[i]), int(semilla[i]), cfg): i for i in range(len(a))}
        cada = max(1, len(a) // 20)
        inicio = time.time()
        for hechas, futuro in enumerate(as_completed(futuros), start=1):
            sigma_t[futuros[futuro]] = futuro.result()
            if hechas % cada == 0 or hechas == len(a):
                print(f'  {nombre}: {hechas}/{len(a)} ({(time.time() - inicio)/60:.1f} min)')
        sigma = sigma_t[:, -1]
        return {'sigma': sigma,
                'log_sigma': np.log10(np.maximum(sigma, SIGMA_MINIMO)),
                'patron': sigma > args.umbral,
                'crece': sigma > sigma_t[:, 0],
                'sigma_t': sigma_t}
    return modelo


def hacer_simulador(rng, executor, cfg, args, nombre):
    import bayesflow as bf
    return bf.simulators.SequentialSimulator([
        bf.simulators.LambdaSimulator(hacer_prior(rng, args), is_batched=True),
        bf.simulators.LambdaSimulator(hacer_modelo(executor, cfg, args, nombre), is_batched=True),
    ])


''' SETTINGS '''

# Cores this process may run on (inside a SLURM job, the allocated ones), not all the cores of the machine
def nucleos_disponibles():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def leer_argumentos():
    parser = argparse.ArgumentParser(description='Simulations on both sides of the Turing threshold (question 4).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--salida', default='simulaciones_existencia', help='output folder')
    parser.add_argument('--entrenar', type=int, default=1000, help='training simulations')
    parser.add_argument('--validar', type=int, default=200, help='validation simulations')
    parser.add_argument('--test', type=int, default=200, help='test simulations')
    parser.add_argument('--semilla', type=int, default=0, help='master seed')
    parser.add_argument('--procesos', type=int, default=nucleos_disponibles(),
                        help='parallel processes (by default, the cores this process may use, e.g. those of a SLURM job)')
    parser.add_argument('--L', type=int, default=128, help='grid side (pixels)')
    parser.add_argument('--t-final', type=float, default=1000.0, help='final time')
    parser.add_argument('--dt-max', type=float, default=0.01, help='largest time step')
    parser.add_argument('--factor-estabilidad', type=float, default=0.2,
                        help='time step = min(dt-max, factor/Dv); Euler is stable up to 0.25')
    parser.add_argument('--ruido', type=float, default=0.01, help='standard deviation of the initial noise')
    parser.add_argument('--a-min', type=float, default=0.1, help='lower bound of a')
    parser.add_argument('--a-max', type=float, default=1.0, help='upper bound of a')
    parser.add_argument('--dv-max', type=float, default=70.0, help='upper bound of Dv (the lower bound is 0)')
    parser.add_argument('--umbral', type=float, default=1e-8,
                        help='label patron: the final standard deviation of u exceeds this value')
    parser.add_argument('--intervalo', type=float, default=10.0, help='time between stored values of sigma(t)')
    args = parser.parse_args()

    if args.entrenar < 1 or args.validar < 1 or args.test < 1:
        parser.error('every set needs at least one simulation')
    if args.L < 3:
        parser.error('--L must be at least 3')
    if args.t_final <= 0 or args.dt_max <= 0 or args.ruido <= 0:
        parser.error('--t-final, --dt-max and --ruido must be > 0')
    if not 0 < args.factor_estabilidad <= 0.25:
        parser.error('--factor-estabilidad must be in (0, 0.25], otherwise Euler blows up')
    if not 0 < args.a_min < args.a_max:
        parser.error('need 0 < --a-min < --a-max')
    if args.dv_max <= 0 or args.umbral <= 0:
        parser.error('--dv-max and --umbral must be > 0')
    if not 0 < args.intervalo <= args.t_final:
        parser.error('--intervalo must be in (0, --t-final]')
    if args.procesos < 1:
        parser.error('--procesos must be at least 1')
    return args


# Every simulation in the (a, Dv) plane: green with a pattern (the initial perturbation has grown), black without
def guardar_patron(datos, ruta):
    a, Dv, crece = datos['a'][:, 0], datos['Dv'][:, 0], datos['crece'][:, 0].astype(bool)
    figura, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(a[crece], Dv[crece], s=8, color='green', label=f'pattern ({crece.sum()})')
    ax.scatter(a[~crece], Dv[~crece], s=8, color='black', label=f'no pattern ({(~crece).sum()})')
    ax.set_xlabel('a')
    ax.set_ylabel('Dv')
    ax.set_title(r'Simulations: pattern if $\sigma(T) > \sigma(0)$')
    ax.legend(loc='upper right')
    figura.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(figura)


''' MAIN '''

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = leer_argumentos()

    if os.path.isdir(args.salida) and any(f.endswith('.npz') for f in os.listdir(args.salida)):
        sys.exit(f'"{args.salida}" already contains simulations; delete it or choose another --salida.')
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, 'parametros.json'), 'w', encoding='utf-8') as archivo:
        json.dump({'tipo': 'existencia', 'sigma_minimo': SIGMA_MINIMO, **vars(args)}, archivo, indent=2)

    num_tiempos = int(round(args.t_final / args.intervalo)) + 1
    tiempos = np.linspace(0, args.intervalo * (num_tiempos - 1), num_tiempos)
    cfg = {'L': args.L, 't_final': args.t_final, 'dt_max': args.dt_max, 'factor_estabilidad': args.factor_estabilidad,
           'ruido': args.ruido, 'tiempos': tiempos.tolist()}
    print(f'{args.entrenar + args.validar + args.test} simulations with {args.procesos} processes -> {args.salida}/')

    inicio = time.time()
    # Each set gets its own random stream, so changing the size of one set does not change the others
    secuencias = np.random.SeedSequence(args.semilla).spawn(len(CONJUNTOS))
    # Workers are started with 'spawn': forking a process that has already imported JAX (through BayesFlow) can deadlock
    with ProcessPoolExecutor(max_workers=args.procesos, mp_context=multiprocessing.get_context('spawn')) as executor:
        for nombre, secuencia in zip(CONJUNTOS, secuencias):
            simulador = hacer_simulador(np.random.default_rng(secuencia), executor, cfg, args, nombre)
            datos = simulador.sample(getattr(args, nombre))
            np.savez(os.path.join(args.salida, f'{nombre}.npz'), tiempos=tiempos, **datos)
            guardar_patron(datos, os.path.join(args.salida, f'patron_{nombre}.png'))
            teoria = datos['Dv'] > D_minimo(datos['a'])
            print(f'{nombre}.npz saved. Agreement with Dv > (3 + 2 sqrt 2)/a: {np.mean(datos["patron"] == teoria):.1%} for '
                  f'sigma(T) > {args.umbral:g}, {np.mean(datos["crece"] == teoria):.1%} for sigma(T) > sigma(0)')
    print(f'Done in {(time.time() - inicio)/60:.1f} min')
