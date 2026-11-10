"""
Are homogeneous a and Dv recovered when the whole pattern is observed?
"""

import os
import sys
import argparse

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bayesflow as bf
import keras


def leer_argumentos():
    parser = argparse.ArgumentParser(description='Inference of homogeneous a and Dv from the whole pattern (question 1).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datos', default='simulaciones_constante', help='folder written by simular_constante.py')
    parser.add_argument('--nombre', default='completo', help='network name')
    parser.add_argument('--epocas', type=int, default=1000, help='training epochs')
    parser.add_argument('--num-entrenar', type=int, default=None, help='use only the first N training simulations')
    parser.add_argument('--lote', type=int, default=32, help='batch size')
    parser.add_argument('--muestras', type=int, default=1000, help='posterior samples per test case')
    parser.add_argument('--redes', default='redes', help='folder for trained networks')
    parser.add_argument('--diagnosticos', default='diagnosticos', help='folder for figures and tables')
    return parser.parse_args()


def cargar(datos, conjunto):
    with np.load(os.path.join(datos, f'{conjunto}.npz')) as archivo:
        return archivo['u'].astype(np.float64), archivo['a'], archivo['Dv']


''' DATA AUGMENTATION '''

# Homogeneous a and Dv on a periodic domain with an isotropic Laplacian: any periodic shift, rotation by 90 degrees
# or reflection of u is another valid simulation with the same parameters. Without it, long training memorizes the
# training set and the posteriors become overconfident
rng_aumentos = np.random.default_rng(0)


def transformar_campo(campo):
    campo = np.roll(campo, tuple(rng_aumentos.integers(campo.shape[0], size=2)), axis=(0, 1))
    campo = np.rot90(campo, rng_aumentos.integers(4))
    return campo[:, ::-1] if rng_aumentos.integers(2) else campo


def aumentar(batch):
    u = np.stack([transformar_campo(campo) for campo in batch["u_final"][..., 0]])
    return {**batch, "u_final": u[..., None]}


''' CALIBRATION TABLE '''

# Well calibrated means coverage ≈ level and mean posterior SD ≈ RMSE. RMSE is sensitive to a few large errors,
# so the median absolute error and the number of cases with |z| > 5 are reported too
NIVELES_COBERTURA = (0.5, 0.9)


def tabla_calibracion(muestras, real):
    mediana = np.median(muestras, axis=1)
    rmse = np.sqrt(np.mean((mediana - real)**2, axis=0))
    error_mediano = np.median(np.abs(mediana - real), axis=0)
    desviacion = muestras.std(axis=1)
    z = (muestras.mean(axis=1) - real) / desviacion
    coberturas = []
    for nivel in NIVELES_COBERTURA:
        inferior, superior = np.quantile(muestras, [(1 - nivel) / 2, (1 + nivel) / 2], axis=1)
        coberturas.append(np.mean((real >= inferior) & (real <= superior), axis=0))

    cabecera = (f'{"var":>3} | {"casos":>5} | {"RMSE":>8} | {"err. mediano":>12} | {"desv. media":>11} | '
                + ' | '.join(f'cobertura {nivel:.0%}' for nivel in NIVELES_COBERTURA) + ' | |z| > 5')
    lineas = [cabecera, '-' * len(cabecera)]
    for idx, variable in enumerate(('a', 'Dv')):
        lineas.append(f'{variable:>3} | {len(real):>5} | {rmse[idx]:>8.4g} | {error_mediano[idx]:>12.4g} | '
                      f'{desviacion[:, idx].mean():>11.4g} | '
                      + ' | '.join(f'{cobertura[idx]:>13.3f}' for cobertura in coberturas)
                      + f' | {np.sum(np.abs(z[:, idx]) > 5):>7d}')
    return '\n'.join(lineas)


''' MAIN '''

if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = leer_argumentos()

    u_entrenar, a_entrenar, Dv_entrenar = cargar(args.datos, 'entrenar')
    if args.num_entrenar is not None:
        if not 1 <= args.num_entrenar <= len(u_entrenar):
            sys.exit(f'--num-entrenar must be between 1 and {len(u_entrenar)}')
        u_entrenar, a_entrenar, Dv_entrenar = u_entrenar[:args.num_entrenar], a_entrenar[:args.num_entrenar], Dv_entrenar[:args.num_entrenar]
    u_validar, a_validar, Dv_validar = cargar(args.datos, 'validar')
    u_test, a_test, Dv_test = cargar(args.datos, 'test')
    print(f'Network "{args.nombre}": {len(u_entrenar)} training, {len(u_validar)} validation and {len(u_test)} test '
          f'simulations of {u_entrenar.shape[1]} x {u_entrenar.shape[2]}; {args.epocas} epochs')

    diccionario_entrenamiento = {"u_final": u_entrenar[..., None], "a": a_entrenar[:, None], "Dv": Dv_entrenar[:, None]}
    diccionario_validar = {"u_final": u_validar[..., None], "a": a_validar[:, None], "Dv": Dv_validar[:, None]}
    diccionario_test = {"u_final": u_test[..., None], "a": a_test[:, None], "Dv": Dv_test[:, None]}

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["a", "Dv"], into="inference_variables")
        .concatenate("u_final", into="summary_variables")
    )

    os.makedirs(args.redes, exist_ok=True)
    ruta_red = os.path.join(args.redes, f'{args.nombre}.keras')
    workflow = bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.ConvolutionalNetwork(summary_dim=16),
        inference_network=bf.networks.FlowMatching(),
        standardize="inference_variables",
        checkpoint_filepath=args.redes,
        checkpoint_name=args.nombre,
        save_best_only=True,
    )

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

    os.makedirs(args.diagnosticos, exist_ok=True)
    figuras = workflow.plot_default_diagnostics(test_data=diccionario_test, variable_keys=["a", "Dv"],
                                                variable_names=[r"$a$", r"$D_v$"])
    for nombre_figura, figura in figuras.items():
        ruta = os.path.join(args.diagnosticos, f'{args.nombre}_{nombre_figura}.png')
        figura.savefig(ruta, dpi=300, bbox_inches='tight')
        plt.close(figura)
        print(f'Saved {ruta}')

    muestras = workflow.sample(conditions={"u_final": u_test[..., None]}, num_samples=args.muestras)
    muestras = np.stack([muestras["a"][..., 0], muestras["Dv"][..., 0]], axis=-1)   # (num_test, muestras, 2)
    tabla = tabla_calibracion(muestras, np.stack([a_test, Dv_test], axis=-1))
    print('\n[test] Calibration\n' + tabla)
    ruta = os.path.join(args.diagnosticos, f'{args.nombre}_calibracion.txt')
    with open(ruta, 'w', encoding='utf-8') as archivo:
        archivo.write(f'Network {args.nombre}, {origen}; {len(u_test)} test simulations\n\n{tabla}\n')
    print(f'Saved {ruta}')
