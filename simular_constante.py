"""
Simulations for Turing patterns with homogeneous a and Dv.
"""

import os
import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# One thread per process: the parallelism comes from the process pool
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import jit

CONJUNTOS = ('entrenar', 'validar', 'test')


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


# Explicit Euler from the u = v = 2 equilibrium plus noise; returns u at the final time
@jit(nopython=True, cache=True)
def simular(a, Dv, L, num_iter, DELTA_T, ruido, semilla):
    np.random.seed(semilla)
    u = 2 + ruido*np.random.standard_normal((L, L))
    v = 2 + ruido*np.random.standard_normal((L, L))
    for _ in range(num_iter):
        u_nuevo = u + DELTA_T*(a*(u + v - 4) - 0.5*(u - 2)**3 + laplaciano(u))
        v_nuevo = v + DELTA_T*(6 - (2*u + v) + Dv*laplaciano(v))
        u, v = u_nuevo, v_nuevo
    return u


# Turing instability threshold: minimum Dv for a given a (Jacobian at u = v = 2 is [[a, a], [-2, -1]])
def D_minimo(a):
    return (3 + 2*np.sqrt(2)) / a


# Explicit Euler is stable if Dv*DELTA_T/dx**2 <= 0.25
def ejecutar(a, Dv, semilla, cfg):
    DELTA_T = min(cfg['dt_max'], cfg['factor_estabilidad'] / Dv)
    num_iter = int(cfg['t_final'] / DELTA_T)
    return simular(a, Dv, cfg['L'], num_iter, DELTA_T, cfg['ruido'], semilla).astype(np.float32)


''' SETTINGS '''

# Cores this process may run on (inside a SLURM job, the allocated ones), not all the cores of the machine
def nucleos_disponibles():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def leer_argumentos():
    parser = argparse.ArgumentParser(description='Simulations with homogeneous a and Dv.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--salida', default='simulaciones_constante', help='output folder')
    parser.add_argument('--entrenar', type=int, default=5000, help='training simulations')
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
    parser.add_argument('--dv-max', type=float, default=70.0, help='upper bound of Dv')
    args = parser.parse_args()

    if args.entrenar < 1 or args.validar < 1 or args.test < 1:
        parser.error('every set needs at least one simulation')
    if args.L < 3:
        parser.error('--L must be at least 3')
    if args.t_final <= 0 or args.dt_max <= 0 or args.ruido < 0:
        parser.error('--t-final and --dt-max must be > 0 and --ruido >= 0')
    if not 0 < args.factor_estabilidad <= 0.25:
        parser.error('--factor-estabilidad must be in (0, 0.25], otherwise Euler blows up')
    if not 0 < args.a_min < args.a_max:
        parser.error('need 0 < --a-min < --a-max')
    if args.dv_max <= D_minimo(args.a_min):
        parser.error(f'--dv-max must exceed the Turing threshold at --a-min ({D_minimo(args.a_min):.4f})')
    if args.procesos < 1:
        parser.error('--procesos must be at least 1')
    return args


# Each set gets its own random stream, so changing the size of one set does not change the others
def muestrear(args):
    muestras = {}
    for nombre, secuencia in zip(CONJUNTOS, np.random.SeedSequence(args.semilla).spawn(len(CONJUNTOS))):
        rng = np.random.default_rng(secuencia)
        num = getattr(args, nombre)
        semillas = rng.integers(2**31, size=num)
        a = rng.uniform(args.a_min, args.a_max, size=num)
        Dv = rng.uniform(D_minimo(a), args.dv_max)
        muestras[nombre] = (a, Dv, semillas)
    return muestras


def guardar_ejemplos(u, a, Dv, ruta, num = 16):
    num = min(num, len(u))
    columnas = int(np.ceil(np.sqrt(num)))
    filas = int(np.ceil(num / columnas))
    figura, axs = plt.subplots(filas, columnas, figsize=(3*columnas, 3*filas), squeeze=False)
    for idx, ax in enumerate(axs.flat):
        ax.axis('off')
        if idx < num:
            ax.imshow(u[idx], cmap='viridis', origin='lower', interpolation='nearest')
            ax.set_title(f'a = {a[idx]:.2f}, Dv = {Dv[idx]:.1f}', fontsize=9)
    figura.savefig(ruta, dpi=100, bbox_inches='tight')
    plt.close(figura)


''' MAIN '''

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = leer_argumentos()

    if os.path.isdir(args.salida) and any(f.endswith('.npz') for f in os.listdir(args.salida)):
        sys.exit(f'"{args.salida}" already contains simulations; delete it or choose another --salida.')
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, 'parametros.json'), 'w', encoding='utf-8') as archivo:
        json.dump({'tipo': 'constante', **vars(args)}, archivo, indent=2)

    cfg = {'L': args.L, 't_final': args.t_final, 'dt_max': args.dt_max,
           'factor_estabilidad': args.factor_estabilidad, 'ruido': args.ruido}
    muestras = muestrear(args)
    print(f'{sum(len(m[0]) for m in muestras.values())} simulations with {args.procesos} processes -> {args.salida}/')

    inicio = time.time()
    with ProcessPoolExecutor(max_workers=args.procesos) as executor:
        for nombre in CONJUNTOS:
            a, Dv, semillas = muestras[nombre]
            u = np.empty((len(a), args.L, args.L), dtype=np.float32)
            futuros = {executor.submit(ejecutar, float(a[i]), float(Dv[i]), int(semillas[i]), cfg): i for i in range(len(a))}
            cada = max(1, len(a) // 20)
            for hechas, futuro in enumerate(as_completed(futuros), start=1):
                u[futuros[futuro]] = futuro.result()
                if hechas % cada == 0 or hechas == len(a):
                    print(f'  {nombre}: {hechas}/{len(a)} ({(time.time() - inicio)/60:.1f} min)')
            np.savez(os.path.join(args.salida, f'{nombre}.npz'), u=u, a=a, Dv=Dv, semilla=semillas)
            guardar_ejemplos(u, a, Dv, os.path.join(args.salida, f'ejemplos_{nombre}.png'))
            print(f'{nombre}.npz saved')
    print(f'Done in {(time.time() - inicio)/60:.1f} min')
