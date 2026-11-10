"""
Question 3 of the paper: are the box partition and the homogeneous a and Dv inside each box recovered from the whole
pattern, without knowing how many boxes there are?

Amortized Bayesian inference (BayesFlow, flow matching) of the a and Dv maps at the resolution of the smallest box
(a k x k grid of cells of side TAM_MINIMO) from the final u of the simulations written by simular_celdas.py. The
partition is then reconstructed bottom-up: four sibling blocks are merged when, for every pair, the credible
interval of the difference of their (a, Dv) contains 0. The credible level is chosen on the validation set and
only reported on the test set.

Training uses data augmentation (rotations by 90 degrees, reflections and periodic shifts by multiples of
TAM_MAXIMO, which preserve the quadtree prior) and keeps the epoch with the lowest validation loss.

Writes:
    <redes>/<nombre>.keras                          trained network (loaded instead of retrained if it exists)
    <diagnosticos>/<nombre>_*.png                   loss, recovery by box size, calibration ECDF (all cells and by
                                                    box size), z-score vs contraction, example partitions
    <diagnosticos>/<nombre>_calibracion.txt         RMSE, median absolute error, posterior SD, coverage by box size
    <diagnosticos>/<nombre>_particiones.txt         partition reconstruction for every credible level
Also prints the test cells with the largest |z-score|.

The defaults are the settings used in the paper:
    python inferir_celdas.py
"""

import os
import sys
import json
import argparse
import itertools

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import bayesflow as bf
import keras

NOMBRES_VARIABLES = [r'$a$', r'$D_v$']


def leer_argumentos():
    parser = argparse.ArgumentParser(description='Inference of the box partition and of a, Dv per box (question 3).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datos', default='simulaciones_celdas', help='folder written by simular_celdas.py')
    parser.add_argument('--nombre', default='celdas', help='network name')
    parser.add_argument('--epocas', type=int, default=1000, help='training epochs')
    parser.add_argument('--num-entrenar', type=int, default=None, help='use only the first N training simulations')
    parser.add_argument('--lote', type=int, default=32, help='batch size')
    parser.add_argument('--muestras', type=int, default=1000, help='posterior samples per test case')
    parser.add_argument('--niveles', default='0.5,0.8,0.9,0.95,0.99,0.995,0.999',
                        help='credible levels tried for the partition reconstruction')
    parser.add_argument('--ejemplos', type=int, default=4, help='test simulations drawn in the partition figure')
    parser.add_argument('--peores', type=int, default=15, help='test cells with the largest |z| that are printed')
    parser.add_argument('--redes', default='redes', help='folder for trained networks')
    parser.add_argument('--diagnosticos', default='diagnosticos', help='folder for figures and tables')
    args = parser.parse_args()
    try:
        args.niveles = [float(nivel) for nivel in args.niveles.split(',')]
    except ValueError:
        parser.error('--niveles must be numbers separated by commas')
    return args


def cargar(datos, conjunto):
    with np.load(os.path.join(datos, f'{conjunto}.npz')) as archivo:
        return archivo['u'].astype(np.float64), archivo['a'], archivo['Dv'], archivo['tam']


''' PARTITIONS '''

# Boxes as (row, column, side) in pixels, recovered from the box-side map (boxes are aligned to their own size)
def cajas_desde_tam(tam, tam_minimo):
    cajas = set()
    for fila, columna in np.ndindex(tam.shape):
        lado = int(tam[fila, columna]) // tam_minimo
        cajas.add((int(fila // lado * lado * tam_minimo), int(columna // lado * lado * tam_minimo), int(tam[fila, columna])))
    return sorted(cajas)


# Boxes (row, column, side, a, Dv) in pixels -> (k, k) maps of a and Dv
def cajas_a_mapas(cajas, k, tam_minimo):
    a, Dv = np.empty((k, k)), np.empty((k, k))
    for fila, columna, lado, a_caja, Dv_caja in cajas:
        celdas = np.s_[fila // tam_minimo:(fila + lado) // tam_minimo, columna // tam_minimo:(columna + lado) // tam_minimo]
        a[celdas], Dv[celdas] = a_caja, Dv_caja
    return a, Dv


# A block stays whole if its 4 children stayed whole and, for every pair, the credible interval of the difference
# contains 0 for both a and Dv. muestras_a, muestras_Dv: (num_muestras, k, k). Returns boxes (row, column, side, a, Dv)
def reconstruir_particion(muestras_a, muestras_Dv, nivel, tam_minimo, tam_maximo):
    k = muestras_a.shape[1]
    bloque_celdas = tam_maximo // tam_minimo
    cola = (1 - nivel) / 2

    def bloque(fila, columna, lado):
        celdas = np.s_[:, fila:fila + lado, columna:columna + lado]
        # Per-sample block mean keeps the posterior correlations
        medias = np.stack([muestras_a[celdas].mean(axis=(1, 2)), muestras_Dv[celdas].mean(axis=(1, 2))], axis=-1)
        caja = [(fila*tam_minimo, columna*tam_minimo, lado*tam_minimo, *np.median(medias, axis=0).tolist())]
        if lado == 1:
            return caja, medias
        mitad = lado // 2
        hijos = [bloque(fila + d_fila, columna + d_columna, mitad) for d_fila in (0, mitad) for d_columna in (0, mitad)]
        cajas_hijos = [caja_hijo for cajas, _ in hijos for caja_hijo in cajas]
        if any(medias_hijo is None for _, medias_hijo in hijos):
            return cajas_hijos, None
        for (_, medias_i), (_, medias_j) in itertools.combinations(hijos, 2):
            inferior, superior = np.quantile(medias_i - medias_j, [cola, 1 - cola], axis=0)
            if np.any((inferior > 0) | (superior < 0)):
                return cajas_hijos, None
        return caja, medias

    return [caja for fila in range(0, k, bloque_celdas) for columna in range(0, k, bloque_celdas)
            for caja in bloque(fila, columna, bloque_celdas)[0]]


def misma_particion(cajas_1, cajas_2):
    return sorted(caja[:3] for caja in cajas_1) == sorted(caja[:3] for caja in cajas_2)


''' MAIN '''

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = leer_argumentos()

    with open(os.path.join(args.datos, 'parametros.json'), encoding='utf-8') as archivo:
        cfg = json.load(archivo)
    TAM_MAXIMO, TAM_MINIMO = cfg['tam_maximo'], cfg['tam_minimo']

    u_entrenar, a_entrenar, Dv_entrenar, _ = cargar(args.datos, 'entrenar')
    if args.num_entrenar is not None:
        if not 1 <= args.num_entrenar <= len(u_entrenar):
            sys.exit(f'--num-entrenar must be between 1 and {len(u_entrenar)}')
        u_entrenar, a_entrenar, Dv_entrenar = u_entrenar[:args.num_entrenar], a_entrenar[:args.num_entrenar], Dv_entrenar[:args.num_entrenar]
    u_validar, a_validar, Dv_validar, tam_validar = cargar(args.datos, 'validar')
    u_test, a_test, Dv_test, tam_test = cargar(args.datos, 'test')
    L = u_entrenar.shape[1]
    k = a_entrenar.shape[1]
    cajas_validar = [cajas_desde_tam(tam, TAM_MINIMO) for tam in tam_validar]
    cajas_test = [cajas_desde_tam(tam, TAM_MINIMO) for tam in tam_test]
    print(f'Network "{args.nombre}": {len(u_entrenar)} training, {len(u_validar)} validation and {len(u_test)} test '
          f'simulations of {L} x {L}; {k} x {k} cells of {TAM_MINIMO} pixels; {args.epocas} epochs')

    # Maps are flattened so cell (i, j) is column i*k + j
    diccionario_entrenamiento = {"u_final": u_entrenar[..., None], "a": a_entrenar.reshape(len(a_entrenar), -1),
                                 "Dv": Dv_entrenar.reshape(len(Dv_entrenar), -1)}
    diccionario_validar = {"u_final": u_validar[..., None], "a": a_validar.reshape(len(a_validar), -1),
                           "Dv": Dv_validar.reshape(len(Dv_validar), -1)}

    ''' DATA AUGMENTATION '''

    # Periodic domain + isotropic Laplacian + shift-invariant prior: rotating, flipping and shifting by multiples of
    # TAM_MAXIMO gives another valid simulation with the same partition prior
    rng_aumentos = np.random.default_rng(0)

    def aumentar(batch):
        num = len(batch["u_final"])
        u = batch["u_final"][..., 0]
        a = batch["a"].reshape(num, k, k)
        Dv = batch["Dv"].reshape(num, k, k)
        u_aum, a_aum, Dv_aum = np.empty_like(u), np.empty_like(a), np.empty_like(Dv)
        for idx in range(num):
            giros = rng_aumentos.integers(4)
            reflejar = rng_aumentos.integers(2)
            desplazamiento = rng_aumentos.integers(L // TAM_MAXIMO, size=2) * TAM_MAXIMO

            def transformar(campo, tam_pixel):
                campo = np.roll(campo, tuple(desplazamiento // tam_pixel), axis=(0, 1))
                campo = np.rot90(campo, giros)
                return campo[:, ::-1] if reflejar else campo

            u_aum[idx] = transformar(u[idx], 1)
            a_aum[idx] = transformar(a[idx], TAM_MINIMO)
            Dv_aum[idx] = transformar(Dv[idx], TAM_MINIMO)
        return {"u_final": u_aum[..., None], "a": a_aum.reshape(num, -1), "Dv": Dv_aum.reshape(num, -1)}

    ''' NETWORK '''

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["a", "Dv"], into="inference_variables")
        .concatenate("u_final", into="summary_variables")
    )
    os.makedirs(args.redes, exist_ok=True)
    ruta_red = os.path.join(args.redes, f'{args.nombre}.keras')
    # pool_head='flatten' keeps where each cell is (global pooling would lose it)
    workflow = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.ConvolutionalNetwork(summary_dim=256, widths=(32, 64, 64, 64), pool_head="flatten"),
        inference_network=bf.networks.FlowMatching(),
        standardize="inference_variables",
        checkpoint_filepath=args.redes,
        checkpoint_name=args.nombre,
        save_best_only=True,
    )

    os.makedirs(args.diagnosticos, exist_ok=True)

    def guardar_figura(figura, nombre_figura):
        ruta = os.path.join(args.diagnosticos, f'{args.nombre}_{nombre_figura}.png')
        figura.savefig(ruta, dpi=300, bbox_inches='tight')
        plt.close(figura)
        print(f'Saved {ruta}')

    if os.path.exists(ruta_red):
        print(f'{ruta_red} exists: loaded, not retrained (delete it to retrain)')
        origen = f'loaded from {ruta_red}'
        workflow.approximator = keras.saving.load_model(ruta_red)
    else:
        history = workflow.fit_offline(data=diccionario_entrenamiento,
                                       epochs=args.epocas,
                                       batch_size=args.lote,
                                       validation_data=diccionario_validar,
                                       augmentations=aumentar)
        # The checkpoint holds the epoch with the lowest validation loss, not the last one
        perdidas = history.history['val_loss']
        mejor = int(np.argmin(perdidas))
        print(f'Best epoch: {mejor + 1} of {args.epocas} (val_loss = {perdidas[mejor]:.4g}, last = {perdidas[-1]:.4g}); loaded')
        workflow.approximator = keras.saving.load_model(ruta_red)
        origen = f'trained on {len(u_entrenar)} simulations from {args.datos}, {args.epocas} epochs, best epoch {mejor + 1}'
        guardar_figura(bf.diagnostics.plots.loss(history), 'losses')

    ''' DIAGNOSTICS '''

    # (num_simul, muestras, k, k) arrays of posterior samples
    def muestrear_mapas(u):
        muestras = workflow.sample(conditions={"u_final": u[..., None]}, num_samples=args.muestras)
        return (muestras["a"].reshape(len(u), args.muestras, k, k), muestras["Dv"].reshape(len(u), args.muestras, k, k))

    muestras_a_validar, muestras_Dv_validar = muestrear_mapas(u_validar)
    muestras_a_test, muestras_Dv_test = muestrear_mapas(u_test)

    # Per-cell recovery on test, split by the size of the box each cell belongs to
    figura, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, muestras, real, nombre in zip(axs, (muestras_a_test, muestras_Dv_test), (a_test, Dv_test), NOMBRES_VARIABLES):
        mediana = np.median(muestras, axis=1)
        for tam in sorted(np.unique(tam_test), reverse=True):
            en_tam = tam_test == tam
            rmse = np.sqrt(np.mean((mediana[en_tam] - real[en_tam])**2))
            ax.scatter(real[en_tam], mediana[en_tam], s=4, alpha=0.4, label=f'boxes of {tam} (RMSE = {rmse:.3g})')
        limites = [real.min(), real.max()]
        ax.plot(limites, limites, 'k--')
        ax.set_title(nombre)
        ax.set_xlabel('Ground truth')
        ax.set_ylabel('Posterior median')
        ax.legend(markerscale=4)
    guardar_figura(figura, 'recovery')

    # All test cells pooled as cases: (num_simul*k*k, muestras, 2). Cells of the same box are not independent,
    # so the ECDF band is narrower than it should be
    muestras_por_celda = np.stack([muestras_a_test, muestras_Dv_test], axis=-1).transpose(0, 2, 3, 1, 4).reshape(-1, args.muestras, 2)
    real_por_celda = np.stack([a_test, Dv_test], axis=-1).reshape(-1, 2)
    tam_por_celda = tam_test.reshape(-1)
    guardar_figura(bf.diagnostics.plots.calibration_ecdf(muestras_por_celda, real_por_celda, variable_names=NOMBRES_VARIABLES), 'calibration_ecdf')
    guardar_figura(bf.diagnostics.plots.z_score_contraction(muestras_por_celda, real_por_celda, variable_names=NOMBRES_VARIABLES), 'z_score_contraction')

    # Calibration by box size. Well calibrated means coverage ≈ level and mean posterior SD ≈ RMSE; RMSE is sensitive
    # to a few large errors, so the median absolute error and the number of cells with |z| > 5 are reported too
    NIVELES_COBERTURA = (0.5, 0.9)
    z = (muestras_por_celda.mean(axis=1) - real_por_celda) / muestras_por_celda.std(axis=1)
    cabecera = (f'{"boxes":>6} | {"var":>3} | {"cells":>6} | {"RMSE":>8} | {"err. mediano":>12} | {"desv. media":>11} | '
                + ' | '.join(f'cobertura {nivel:.0%}' for nivel in NIVELES_COBERTURA) + ' | |z| > 5')
    lineas = [cabecera, '-' * len(cabecera)]
    grupos = [(f'{tam}', tam_por_celda == tam) for tam in sorted(np.unique(tam_por_celda), reverse=True)]
    grupos.append(('all', np.ones_like(tam_por_celda, dtype=bool)))
    for nombre_grupo, en_grupo in grupos:
        muestras_grupo, real_grupo = muestras_por_celda[en_grupo], real_por_celda[en_grupo]
        mediana = np.median(muestras_grupo, axis=1)
        rmse = np.sqrt(np.mean((mediana - real_grupo)**2, axis=0))
        error_mediano = np.median(np.abs(mediana - real_grupo), axis=0)
        desviacion = muestras_grupo.std(axis=1).mean(axis=0)
        coberturas = []
        for nivel in NIVELES_COBERTURA:
            inferior, superior = np.quantile(muestras_grupo, [(1 - nivel) / 2, (1 + nivel) / 2], axis=1)
            coberturas.append(np.mean((real_grupo >= inferior) & (real_grupo <= superior), axis=0))
        for idx_var, variable in enumerate(('a', 'Dv')):
            lineas.append(f'{nombre_grupo:>6} | {variable:>3} | {en_grupo.sum():>6} | {rmse[idx_var]:>8.4g} | '
                          f'{error_mediano[idx_var]:>12.4g} | {desviacion[idx_var]:>11.4g} | '
                          + ' | '.join(f'{cobertura[idx_var]:>13.3f}' for cobertura in coberturas)
                          + f' | {np.sum(np.abs(z[en_grupo, idx_var]) > 5):>7d}')
        if nombre_grupo != 'all':
            figura = bf.diagnostics.plots.calibration_ecdf(muestras_grupo, real_grupo, variable_names=NOMBRES_VARIABLES)
            figura.suptitle(f'Boxes of {nombre_grupo} pixels ({en_grupo.sum()} test cells)', y=1.03)
            guardar_figura(figura, f'calibration_ecdf_boxes_{nombre_grupo}')
    tabla = '\n'.join(lineas)
    print('\n[test] Calibration by box size\n' + tabla)
    ruta = os.path.join(args.diagnosticos, f'{args.nombre}_calibracion.txt')
    with open(ruta, 'w', encoding='utf-8') as archivo:
        archivo.write(f'Network {args.nombre}, {origen}; {len(u_test)} test simulations\n\n{tabla}\n')
    print(f'Saved {ruta}')

    # Cells where the network is most confidently wrong; z = (posterior mean - truth) / posterior SD
    idx_simulacion, idx_fila, idx_columna = np.unravel_index(np.arange(len(real_por_celda)), (len(u_test), k, k))
    print(f'\n[test] {args.peores} cells with the largest |z| (test simulation numbered from 0, cell row,column)')
    print(f'{"var":>3} | {"simul":>5} | {"cell":>5} | {"box":>4} | {"a true":>7} | {"Dv true":>7} | {"median":>8} | {"sd":>7} | {"z":>7}')
    for idx_var, variable in enumerate(('a', 'Dv')):
        for celda in np.argsort(-np.abs(z[:, idx_var]))[:args.peores]:
            print(f'{variable:>3} | {idx_simulacion[celda]:>5} | {idx_fila[celda]:>2},{idx_columna[celda]:<2} | {tam_por_celda[celda]:>4} | '
                  f'{real_por_celda[celda, 0]:>7.3f} | {real_por_celda[celda, 1]:>7.2f} | '
                  f'{np.median(muestras_por_celda[celda, :, idx_var]):>8.3f} | {muestras_por_celda[celda, :, idx_var].std():>7.3f} | '
                  f'{z[celda, idx_var]:>7.2f}')

    ''' PARTITION RECONSTRUCTION '''

    # Rand index only over pairs inside the same TAM_MAXIMO block: pairs across blocks always agree and would inflate it
    bloque_celdas = TAM_MAXIMO // TAM_MINIMO
    fila_celda, columna_celda = np.divmod(np.arange(k*k), k)
    bloque_celda = (fila_celda // bloque_celdas) * (k // bloque_celdas) + columna_celda // bloque_celdas
    celda_i, celda_j = np.triu_indices(k*k, 1)
    en_mismo_bloque = bloque_celda[celda_i] == bloque_celda[celda_j]
    pares_i, pares_j = celda_i[en_mismo_bloque], celda_j[en_mismo_bloque]

    def etiquetas_celdas(cajas):
        etiquetas = np.empty((k, k), dtype=int)
        for idx, (fila, columna, lado, *_) in enumerate(cajas):
            etiquetas[fila // TAM_MINIMO:(fila + lado) // TAM_MINIMO, columna // TAM_MINIMO:(columna + lado) // TAM_MINIMO] = idx
        return etiquetas.ravel()

    def indice_rand(cajas_1, cajas_2):
        etiquetas_1, etiquetas_2 = etiquetas_celdas(cajas_1), etiquetas_celdas(cajas_2)
        return np.mean((etiquetas_1[pares_i] == etiquetas_1[pares_j]) == (etiquetas_2[pares_i] == etiquetas_2[pares_j]))

    # Returns (fraction of exact partitions, mean Rand index) over a set
    def evaluar_particiones(muestras_a, muestras_Dv, lista_cajas_reales, nivel):
        reconstruidas = [reconstruir_particion(muestras_a[i], muestras_Dv[i], nivel, TAM_MINIMO, TAM_MAXIMO)
                         for i in range(len(lista_cajas_reales))]
        exactas = np.mean([misma_particion(rec, real) for rec, real in zip(reconstruidas, lista_cajas_reales)])
        rand = np.mean([indice_rand(rec, real) for rec, real in zip(reconstruidas, lista_cajas_reales)])
        return exactas, rand

    # The level is chosen on validation and only then evaluated on test, so the test numbers are not optimistic
    lineas = ['[validation] level | exact partitions | mean Rand index (pairs in the same block)']
    resultados = []
    for nivel in args.niveles:
        exactas, rand = evaluar_particiones(muestras_a_validar, muestras_Dv_validar, cajas_validar, nivel)
        resultados.append((exactas, rand, nivel))
        lineas.append(f'{nivel:>18} | {exactas:>16.3f} | {rand:.3f}')
    nivel_elegido = max(resultados)[2]   # most exact partitions first, Rand index breaks ties
    exactas_test, rand_test = evaluar_particiones(muestras_a_test, muestras_Dv_test, cajas_test, nivel_elegido)
    lineas.append(f'Level chosen on validation: {nivel_elegido}')
    lineas.append(f'[test] level {nivel_elegido}: exact partitions = {exactas_test:.3f}, mean Rand index = {rand_test:.3f}')
    texto = '\n'.join(lineas)
    print('\n' + texto)
    ruta = os.path.join(args.diagnosticos, f'{args.nombre}_particiones.txt')
    with open(ruta, 'w', encoding='utf-8') as archivo:
        archivo.write(f'Network {args.nombre}, {origen}\n\n{texto}\n')
    print(f'Saved {ruta}')

    # Example partitions: u, true and reconstructed a and Dv, and posterior SD, with the boxes drawn on top
    num = min(args.ejemplos, len(u_test))
    figura, axs = plt.subplots(num, 7, figsize=(24, 3.4*num), squeeze=False)
    extension = (-0.5, L - 0.5, -0.5, L - 0.5)
    for idx, fila_ax in enumerate(axs):
        cajas_rec = reconstruir_particion(muestras_a_test[idx], muestras_Dv_test[idx], nivel_elegido, TAM_MINIMO, TAM_MAXIMO)
        a_rec, Dv_rec = cajas_a_mapas(cajas_rec, k, TAM_MINIMO)
        paneles = [(u_test[idx], 'u', cajas_test[idx], None),
                   (a_test[idx], 'a true', cajas_test[idx], a_test[idx]),
                   (a_rec, f'a reconstructed ({len(cajas_rec)} boxes)', cajas_rec, a_test[idx]),
                   (muestras_a_test[idx].std(axis=0), 'posterior SD of a', cajas_rec, None),
                   (Dv_test[idx], 'Dv true', cajas_test[idx], Dv_test[idx]),
                   (Dv_rec, f'Dv reconstructed ({len(cajas_rec)} boxes)', cajas_rec, Dv_test[idx]),
                   (muestras_Dv_test[idx].std(axis=0), 'posterior SD of Dv', cajas_rec, None)]
        for ax, (campo, titulo, cajas, referencia) in zip(fila_ax, paneles):
            limites = {} if referencia is None else {'vmin': referencia.min(), 'vmax': referencia.max()}
            im = ax.imshow(campo, cmap='viridis', origin='lower', interpolation='nearest', extent=extension, **limites)
            for fila, columna, lado, *_ in cajas:
                ax.add_patch(Rectangle((columna - 0.5, fila - 0.5), lado, lado, fill=False, edgecolor='white', linewidth=0.8))
            ax.set_title(titulo)
            ax.set_xticks([])
            ax.set_yticks([])
            figura.colorbar(im, ax=ax, shrink=0.8)
    guardar_figura(figura, 'particiones_test')
