"""
Simulations for Turing patterns with a and Dv constant on the boxes of a random quadtree.
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
from matplotlib.patches import Rectangle
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


# div(D grad c) with spatially varying D, periodic; uses the mean of D on each face so mass is conserved
@jit(nopython=True, cache=True)
def divergencia_difusion(matriz, D):
    n, m = matriz.shape
    div = np.empty_like(matriz)
    for i in range(n):
        for j in range(m):
            c = matriz[i, j]
            Dc = D[i, j]
            ip, im, jp, jm = (i+1) % n, (i-1) % n, (j+1) % m, (j-1) % m
            div[i, j] = ((Dc + D[ip, j])*(matriz[ip, j] - c) + (Dc + D[im, j])*(matriz[im, j] - c)
                         + (Dc + D[i, jp])*(matriz[i, jp] - c) + (Dc + D[i, jm])*(matriz[i, jm] - c)) * 0.5
    return div


# Explicit Euler from the u = v = 2 equilibrium plus noise; a and Dv are (L, L) maps; returns u at the final time
@jit(nopython=True, cache=True)
def simular(a, Dv, L, num_iter, DELTA_T, ruido, semilla):
    np.random.seed(semilla)
    u = 2 + ruido*np.random.standard_normal((L, L))
    v = 2 + ruido*np.random.standard_normal((L, L))
    for _ in range(num_iter):
        u_nuevo = u + DELTA_T*(a*(u + v - 4) - 0.5*(u - 2)**3 + laplaciano(u))
        v_nuevo = v + DELTA_T*(6 - (2*u + v) + divergencia_difusion(v, Dv))
        u, v = u_nuevo, v_nuevo
    return u


# Turing instability threshold: minimum Dv for a given a (Jacobian at u = v = 2 is [[a, a], [-2, -1]])
def D_minimo(a):
    return (3 + 2*np.sqrt(2)) / a


# Cell maps (k, k) -> pixel maps (L, L)
def a_pixeles(mapa, tam_minimo):
    return np.repeat(np.repeat(mapa, tam_minimo, axis=0), tam_minimo, axis=1)


# Explicit Euler is stable if max(Dv)*DELTA_T/dx**2 <= 0.25
def ejecutar(a_celdas, Dv_celdas, semilla, cfg):
    a = a_pixeles(a_celdas, cfg['tam_minimo'])
    Dv = a_pixeles(Dv_celdas, cfg['tam_minimo'])
    DELTA_T = min(cfg['dt_max'], cfg['factor_estabilidad'] / Dv.max())
    num_iter = int(cfg['t_final'] / DELTA_T)
    return simular(a, Dv, cfg['L'], num_iter, DELTA_T, cfg['ruido'], semilla).astype(np.float32)


''' PARTITION '''

# Random quadtree on a k x k grid of cells; returns the a, Dv and box-side (pixels) maps per cell
def muestrear_particion(rng, args):
    k = args.L // args.tam_minimo
    a = np.empty((k, k))
    Dv = np.empty((k, k))
    tam = np.empty((k, k), dtype=np.int32)

    def partir(fila, columna, lado):
        if lado > 1 and rng.random() < args.prob_partir:
            mitad = lado // 2
            for d_fila in (0, mitad):
                for d_columna in (0, mitad):
                    partir(fila + d_fila, columna + d_columna, mitad)
        else:
            a_caja = rng.uniform(args.a_min, args.a_max)
            celdas = np.s_[fila:fila + lado, columna:columna + lado]
            a[celdas] = a_caja
            Dv[celdas] = rng.uniform(D_minimo(a_caja), args.dv_max)
            tam[celdas] = lado * args.tam_minimo

    bloque = args.tam_maximo // args.tam_minimo
    for fila in range(0, k, bloque):
        for columna in range(0, k, bloque):
            partir(fila, columna, bloque)
    return a, Dv, tam


# Boxes as (row, column, side) in pixels, recovered from the box-side map (boxes are aligned to their own size)
def cajas_desde_tam(tam, tam_minimo):
    cajas = set()
    for fila, columna in np.ndindex(tam.shape):
        lado = int(tam[fila, columna]) // tam_minimo
        cajas.add((int(fila // lado * lado * tam_minimo), int(columna // lado * lado * tam_minimo), int(tam[fila, columna])))
    return sorted(cajas)


''' SETTINGS '''

# Cores this process may run on (inside a SLURM job, the allocated ones), not all the cores of the machine
def nucleos_disponibles():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def es_potencia_de_2(n):
    return n > 0 and (n & (n - 1)) == 0


def leer_argumentos():
    parser = argparse.ArgumentParser(description='Simulations with a and Dv constant on quadtree boxes.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--salida', default='simulaciones_celdas', help='output folder')
    parser.add_argument('--entrenar', type=int, default=5000, help='training simulations')
    parser.add_argument('--validar', type=int, default=200, help='validation simulations')
    parser.add_argument('--test', type=int, default=200, help='test simulations')
    parser.add_argument('--semilla', type=int, default=0, help='master seed')
    parser.add_argument('--procesos', type=int, default=nucleos_disponibles(),
                        help='parallel processes (by default, the cores this process may use, e.g. those of a SLURM job)')
    parser.add_argument('--L', type=int, default=128, help='grid side (pixels); multiple of --tam-maximo')
    parser.add_argument('--tam-maximo', type=int, default=64, help='side of the largest boxes (pixels)')
    parser.add_argument('--tam-minimo', type=int, default=16, help='side of the smallest boxes (pixels)')
    parser.add_argument('--prob-partir', type=float, default=0.5, help='probability of splitting a box into four')
    parser.add_argument('--t-final', type=float, default=1000.0, help='final time')
    parser.add_argument('--dt-max', type=float, default=0.01, help='largest time step')
    parser.add_argument('--factor-estabilidad', type=float, default=0.2,
                        help='time step = min(dt-max, factor/max(Dv)); Euler is stable up to 0.25')
    parser.add_argument('--ruido', type=float, default=0.01, help='standard deviation of the initial noise')
    parser.add_argument('--a-min', type=float, default=0.1, help='lower bound of a')
    parser.add_argument('--a-max', type=float, default=1.0, help='upper bound of a')
    parser.add_argument('--dv-max', type=float, default=70.0, help='upper bound of Dv')
    args = parser.parse_args()

    if args.entrenar < 1 or args.validar < 1 or args.test < 1:
        parser.error('every set needs at least one simulation')
    if args.tam_minimo < 1 or args.tam_maximo % args.tam_minimo != 0 or not es_potencia_de_2(args.tam_maximo // args.tam_minimo):
        parser.error('--tam-maximo must be --tam-minimo times 1, 2, 4, 8...')
    if args.L < args.tam_maximo or args.L % args.tam_maximo != 0:
        parser.error('--L must be a multiple of --tam-maximo')
    if not 0 <= args.prob_partir <= 1:
        parser.error('--prob-partir must be in [0, 1]')
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
        particiones = [muestrear_particion(rng, args) for _ in range(num)]
        a, Dv, tam = (np.stack(mapas) for mapas in zip(*particiones))
        muestras[nombre] = (a, Dv, tam, semillas)
    return muestras


def guardar_ejemplos(u, a, Dv, tam, tam_minimo, ruta, num = 4):
    num = min(num, len(u))
    figura, axs = plt.subplots(num, 3, figsize=(12, 4*num), squeeze=False)
    extension = (-0.5, u.shape[1] - 0.5, -0.5, u.shape[1] - 0.5)
    for fila_ax, idx in zip(axs, range(num)):
        cajas = cajas_desde_tam(tam[idx], tam_minimo)
        for ax, campo, titulo in zip(fila_ax, (u[idx], a[idx], Dv[idx]), ('u', 'a', 'Dv')):
            im = ax.imshow(campo, cmap='viridis', origin='lower', interpolation='nearest', extent=extension)
            for fila, columna, lado in cajas:
                ax.add_patch(Rectangle((columna - 0.5, fila - 0.5), lado, lado, fill=False, edgecolor='white', linewidth=0.8))
            ax.set_title(f'{titulo} ({len(cajas)} cajas)')
            ax.set_xticks([])
            ax.set_yticks([])
            figura.colorbar(im, ax=ax, shrink=0.8)
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
        json.dump({'tipo': 'celdas', **vars(args)}, archivo, indent=2)

    cfg = {'L': args.L, 'tam_minimo': args.tam_minimo, 't_final': args.t_final, 'dt_max': args.dt_max,
           'factor_estabilidad': args.factor_estabilidad, 'ruido': args.ruido}
    muestras = muestrear(args)
    print(f'{sum(len(m[0]) for m in muestras.values())} simulations with {args.procesos} processes -> {args.salida}/')

    inicio = time.time()
    with ProcessPoolExecutor(max_workers=args.procesos) as executor:
        for nombre in CONJUNTOS:
            a, Dv, tam, semillas = muestras[nombre]
            u = np.empty((len(a), args.L, args.L), dtype=np.float32)
            futuros = {executor.submit(ejecutar, a[i], Dv[i], int(semillas[i]), cfg): i for i in range(len(a))}
            cada = max(1, len(a) // 20)
            for hechas, futuro in enumerate(as_completed(futuros), start=1):
                u[futuros[futuro]] = futuro.result()
                if hechas % cada == 0 or hechas == len(a):
                    print(f'  {nombre}: {hechas}/{len(a)} ({(time.time() - inicio)/60:.1f} min)')
            np.savez(os.path.join(args.salida, f'{nombre}.npz'), u=u, a=a, Dv=Dv, tam=tam, semilla=semillas)
            guardar_ejemplos(u, a, Dv, tam, args.tam_minimo, os.path.join(args.salida, f'ejemplos_{nombre}.png'))
            print(f'{nombre}.npz saved')
    print(f'Done in {(time.time() - inicio)/60:.1f} min')
