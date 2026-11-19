"""
Are homogeneous a and Dv recovered when only part of the pattern is observed?
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

# Validation is not augmented and uses the training batch size, so it keeps at most this many random windows per
# simulation (otherwise validating the small windows would take longer than training)
VENTANAS_VALIDACION = 16


def leer_argumentos():
    parser = argparse.ArgumentParser(description='Inference of homogeneous a and Dv from a window of the pattern (question 2).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datos', default='simulaciones_constante', help='folder written by simular_constante.py')
    parser.add_argument('--ventanas', default='64,32,16,8,4,2', help='window sides, comma separated')
    parser.add_argument('--nombre', default='ventanas', help='network name (the window size is appended)')
    parser.add_argument('--epocas', type=int, default=1000, help='training epochs')
    parser.add_argument('--num-entrenar', type=int, default=None, help='use only the first N training simulations')
    parser.add_argument('--lote', type=int, default=32, help='simulations per batch (each one is cut into all its windows)')
    parser.add_argument('--muestras', type=int, default=1000, help='posterior samples per test case')
    parser.add_argument('--redes', default='redes', help='folder for trained networks')
    parser.add_argument('--diagnosticos', default='diagnosticos', help='folder for figures and tables')
    args = parser.parse_args()
    try:
        args.ventanas = [int(tam) for tam in args.ventanas.split(',')]
    except ValueError:
        parser.error('--ventanas must be integers separated by commas, e.g. 64,32,16')
    return args


def cargar(datos, conjunto):
    with np.load(os.path.join(datos, f'{conjunto}.npz')) as archivo:
        return archivo['u'].astype(np.float64), archivo['a'], archivo['Dv']


# Non-overlapping windows; the k*k windows of each simulation stay consecutive and share its a and Dv
def partir_en_ventanas(u, a, Dv, tam_ventana):
    num_simul, L, _ = u.shape
    k = L // tam_ventana
    ventanas = u.reshape(num_simul, k, tam_ventana, k, tam_ventana).transpose(0, 1, 3, 2, 4)
    ventanas = ventanas.reshape(num_simul * k * k, tam_ventana, tam_ventana)
    return ventanas, np.repeat(a, k * k), np.repeat(Dv, k * k)


''' DATA AUGMENTATION '''

# Homogeneous a and Dv on a periodic domain with an isotropic Laplacian: any periodic shift, rotation by 90 degrees
# or reflection of u is another valid simulation with the same parameters
rng_aumentos = np.random.default_rng(0)


def transformar_campo(campo):
    campo = np.roll(campo, tuple(rng_aumentos.integers(campo.shape[0], size=2)), axis=(0, 1))
    campo = np.rot90(campo, rng_aumentos.integers(4))
    return campo[:, ::-1] if rng_aumentos.integers(2) else campo


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
    L = u_entrenar.shape[1]
    for tam in args.ventanas:
        if tam < 1 or L % tam != 0:
            sys.exit(f'Window side {tam} does not divide L = {L}')
    print(f'{len(u_entrenar)} training, {len(u_validar)} validation and {len(u_test)} test simulations of {L} x {L}; '
          f'windows {args.ventanas}; {args.epocas} epochs')

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["a", "Dv"], into="inference_variables")
        .concatenate("u_final", into="summary_variables")
    )
    os.makedirs(args.redes, exist_ok=True)
    os.makedirs(args.diagnosticos, exist_ok=True)
    # Training data are the full fields; the augmentation transforms them and cuts them into windows
    diccionario_entrenamiento = {"u_final": u_entrenar[..., None], "a": a_entrenar[:, None], "Dv": Dv_entrenar[:, None]}

    for tam_ventana in args.ventanas:
        nombre = f'{args.nombre}_{tam_ventana}x{tam_ventana}'
        ventanas_por_simul = (L // tam_ventana)**2
        print(f'\n===== {nombre}: {ventanas_por_simul} windows per simulation =====')

        def aumentar(batch, tam_ventana=tam_ventana):
            u = np.stack([transformar_campo(campo) for campo in batch["u_final"][..., 0]])
            u_vent, a_vent, Dv_vent = partir_en_ventanas(u, batch["a"][:, 0], batch["Dv"][:, 0], tam_ventana)
            return {"u_final": u_vent[..., None], "a": a_vent[:, None], "Dv": Dv_vent[:, None]}

        u_validar_vent, a_validar_vent, Dv_validar_vent = partir_en_ventanas(u_validar, a_validar, Dv_validar, tam_ventana)
        rng_validacion = np.random.default_rng(1)
        por_simul = min(ventanas_por_simul, VENTANAS_VALIDACION)
        idx_validacion = np.concatenate([i * ventanas_por_simul + rng_validacion.choice(ventanas_por_simul, por_simul, replace=False)
                                         for i in range(len(u_validar))])
        diccionario_validar = {"u_final": u_validar_vent[idx_validacion, ..., None], "a": a_validar_vent[idx_validacion, None],
                               "Dv": Dv_validar_vent[idx_validacion, None]}

        # First window of each test simulation
        u_test_vent = u_test[:, :tam_ventana, :tam_ventana]
        diccionario_test = {"u_final": u_test_vent[..., None], "a": a_test[:, None], "Dv": Dv_test[:, None]}

        ruta_red = os.path.join(args.redes, f'{nombre}.keras')
        workflow = bf.BasicWorkflow(
            adapter=adapter,
            summary_network=bf.networks.ConvolutionalNetwork(summary_dim=16),
            inference_network=bf.networks.FlowMatching(),
            standardize="inference_variables",
            checkpoint_filepath=args.redes,
            checkpoint_name=nombre,
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

        figuras = workflow.plot_default_diagnostics(test_data=diccionario_test, variable_keys=["a", "Dv"],
                                                    variable_names=[r"$a$", r"$D_v$"])
        for nombre_figura, figura in figuras.items():
            ruta = os.path.join(args.diagnosticos, f'{nombre}_{nombre_figura}.png')
            figura.savefig(ruta, dpi=300, bbox_inches='tight')
            plt.close(figura)
            print(f'Saved {ruta}')

        muestras = workflow.sample(conditions={"u_final": u_test_vent[..., None]}, num_samples=args.muestras)
        muestras = np.stack([muestras["a"][..., 0], muestras["Dv"][..., 0]], axis=-1)   # (num_test, muestras, 2)
        tabla = tabla_calibracion(muestras, np.stack([a_test, Dv_test], axis=-1))
        print(f'\n[test] Calibration, {nombre}\n' + tabla)
        ruta = os.path.join(args.diagnosticos, f'{nombre}_calibracion.txt')
        with open(ruta, 'w', encoding='utf-8') as archivo:
            archivo.write(f'Network {nombre}, {origen}; first window of each of the {len(u_test)} test simulations\n\n{tabla}\n')
        print(f'Saved {ruta}')
