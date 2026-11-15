"""
Question 4 of the paper: can the condition for the existence of a Turing pattern be learned from simulations?

A BayesFlow network (flow matching) learns the distribution of log10 sigma(T) given (a, Dv), where sigma(T) is the
final spatial standard deviation of u in the simulations written by simular_existencia.py (amortized likelihood
estimation: (a, Dv) are the conditions and log10 sigma the variable whose distribution is learned). The epoch with the
lowest validation loss is kept.

After the usual diagnostics on the test set, the network is evaluated on PARES new (a, Dv) pairs drawn uniformly over
the simulated range: for each pair, the probability of a pattern is the fraction of sampled log10 sigma(T) above
log10 sigma(0) (the initial perturbation has grown), and the pair is labelled as having a pattern when that probability
is at least 0.5. The labelled pairs are drawn like the simulations (green with a pattern, black without) together with
the theoretical Turing threshold Dv = (3 + 2 sqrt(2)) / a. The boundary is also summarised by the constant c of the
curve Dv = c / a that best separates the predicted labels, to compare with the theoretical c = 3 + 2 sqrt(2) = 5.83.

Writes:
    <redes>/<nombre>.keras                       trained network (loaded instead of retrained if it exists)
    <diagnosticos>/<nombre>_losses.png           training and validation loss
    <diagnosticos>/<nombre>_recovery.png         predicted vs simulated log10 sigma(T) on the test set
    <diagnosticos>/<nombre>_calibration_ecdf.png calibration of the predicted distribution on the test set
    <diagnosticos>/<nombre>_patron.png           the PARES pairs, green with a pattern and black without, with the theory
    <diagnosticos>/<nombre>_resultados.txt       classification of the test set and of the pairs, fitted constant c
    <diagnosticos>/<nombre>_pares.npz            the pairs, their probability of a pattern and median log10 sigma(T)

The defaults are the settings used in the paper:
    python inferir_existencia.py
"""

import os
import sys
import json
import argparse

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bayesflow as bf
import keras

C_TEORICO = 3 + 2*np.sqrt(2)


def leer_argumentos():
    parser = argparse.ArgumentParser(description='Learning the condition for pattern existence (question 4).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datos', default='simulaciones_existencia', help='folder written by simular_existencia.py')
    parser.add_argument('--nombre', default='existencia', help='network name')
    parser.add_argument('--epocas', type=int, default=1000, help='training epochs')
    parser.add_argument('--num-entrenar', type=int, default=None, help='use only the first N training simulations')
    parser.add_argument('--lote', type=int, default=32, help='batch size')
    parser.add_argument('--muestras', type=int, default=1000, help='samples of log10 sigma per test case')
    parser.add_argument('--pares', type=int, default=10_000, help='new (a, Dv) pairs classified with the network')
    parser.add_argument('--muestras-pares', type=int, default=200, help='samples of log10 sigma per pair')
    parser.add_argument('--semilla-pares', type=int, default=0, help='seed of the new (a, Dv) pairs')
    parser.add_argument('--redes', default='redes', help='folder for trained networks')
    parser.add_argument('--diagnosticos', default='diagnosticos', help='folder for figures and tables')
    return parser.parse_args()


def cargar(datos, conjunto):
    with np.load(os.path.join(datos, f'{conjunto}.npz')) as archivo:
        return {clave: archivo[clave].astype(np.float64) for clave in ('a', 'Dv', 'log_sigma', 'patron', 'crece')}


# Constant c of the curve Dv = c / a that best separates the labels (pattern above the curve, none below)
def ajustar_c(a, Dv, patron):
    candidatos = np.sort(a * Dv)
    aciertos = [np.mean((Dv > c / a) == patron) for c in candidatos]
    mejor = int(np.argmax(aciertos))
    return candidatos[mejor], aciertos[mejor]


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = leer_argumentos()

    with open(os.path.join(args.datos, 'parametros.json'), encoding='utf-8') as archivo:
        cfg = json.load(archivo)
    # A pattern means the initial perturbation has grown: sigma(T) > sigma(0), the standard deviation of the initial noise
    LOG_INICIAL = np.log10(cfg['ruido'])

    entrenar = cargar(args.datos, 'entrenar')
    if args.num_entrenar is not None:
        if not 1 <= args.num_entrenar <= len(entrenar['a']):
            sys.exit(f'--num-entrenar must be between 1 and {len(entrenar["a"])}')
        entrenar = {clave: valor[:args.num_entrenar] for clave, valor in entrenar.items()}
    validar = cargar(args.datos, 'validar')
    test = cargar(args.datos, 'test')
    print(f'Network "{args.nombre}": {len(entrenar["a"])} training, {len(validar["a"])} validation and '
          f'{len(test["a"])} test simulations; {args.epocas} epochs')

    variables = ('a', 'Dv', 'log_sigma')
    diccionario_entrenamiento = {clave: entrenar[clave] for clave in variables}
    diccionario_validar = {clave: validar[clave] for clave in variables}

    ''' NETWORK '''

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["a", "Dv"], into="inference_conditions")
        .concatenate(["log_sigma"], into="inference_variables")
    )
    os.makedirs(args.redes, exist_ok=True)
    ruta_red = os.path.join(args.redes, f'{args.nombre}.keras')
    workflow = bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(),
        standardize=["inference_variables", "inference_conditions"],
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
        workflow.approximator = keras.saving.load_model(ruta_red)
        origen = f'loaded from {ruta_red}'
    else:
        history = workflow.fit_offline(data=diccionario_entrenamiento,
                                       epochs=args.epocas,
                                       batch_size=args.lote,
                                       validation_data=diccionario_validar)
        # The checkpoint holds the epoch with the lowest validation loss, not the last one
        perdidas = history.history['val_loss']
        mejor = int(np.argmin(perdidas))
        print(f'Best epoch: {mejor + 1} of {args.epocas} (val_loss = {perdidas[mejor]:.4g}, last = {perdidas[-1]:.4g}); loaded')
        workflow.approximator = keras.saving.load_model(ruta_red)
        origen = f'trained on {len(entrenar["a"])} simulations from {args.datos}, {args.epocas} epochs, best epoch {mejor + 1}'
        guardar_figura(bf.diagnostics.plots.loss(history), 'losses')

    ''' DIAGNOSTICS ON THE TEST SET '''

    muestras = workflow.sample(conditions={"a": test["a"], "Dv": test["Dv"]}, num_samples=args.muestras)["log_sigma"]
    nombres = [r'$\log_{10}\,\sigma(T)$']
    guardar_figura(bf.diagnostics.plots.recovery(muestras, test["log_sigma"], variable_names=nombres), 'recovery')
    guardar_figura(bf.diagnostics.plots.calibration_ecdf(muestras, test["log_sigma"], variable_names=nombres), 'calibration_ecdf')

    prediccion_test = np.mean(muestras[:, :, 0] > LOG_INICIAL, axis=1) >= 0.5
    teoria_test = test["Dv"][:, 0] > C_TEORICO / test["a"][:, 0]
    lineas = [f'Network {args.nombre}, {origen}', '',
              f'Test set ({len(test["a"])} simulations): pattern if P(sigma(T) > sigma(0)) >= 0.5',
              f'  agreement with the simulated label sigma(T) > sigma(0): {np.mean(prediccion_test == test["crece"][:, 0].astype(bool)):.1%}',
              f'  agreement with the theory Dv > (3 + 2 sqrt 2)/a:         {np.mean(prediccion_test == teoria_test):.1%}']

    ''' NEW (a, Dv) PAIRS '''

    rng = np.random.default_rng(args.semilla_pares)
    a_pares = rng.uniform(cfg['a_min'], cfg['a_max'], size=(args.pares, 1))
    Dv_pares = rng.uniform(0, cfg['dv_max'], size=(args.pares, 1))
    muestras_pares = workflow.sample(conditions={"a": a_pares, "Dv": Dv_pares}, num_samples=args.muestras_pares,
                                     batch_size=1000)["log_sigma"][..., 0]
    a_pares, Dv_pares = a_pares[:, 0], Dv_pares[:, 0]
    probabilidad = np.mean(muestras_pares > LOG_INICIAL, axis=1)
    patron = probabilidad >= 0.5
    teoria = Dv_pares > C_TEORICO / a_pares
    c_aprendido, acierto_c = ajustar_c(a_pares, Dv_pares, patron)
    lineas += ['', f'{args.pares} new (a, Dv) pairs, a ~ U({cfg["a_min"]}, {cfg["a_max"]}), Dv ~ U(0, {cfg["dv_max"]}): '
                   f'pattern if P(sigma(T) > sigma(0)) >= 0.5',
               f'  with a pattern: {patron.mean():.1%} (theory: {teoria.mean():.1%})',
               f'  agreement with the theory Dv > (3 + 2 sqrt 2)/a: {np.mean(patron == teoria):.2%}',
               f'  curve Dv = c / a that best separates the predicted labels: c = {c_aprendido:.4f} '
               f'(theory {C_TEORICO:.4f}, {c_aprendido / C_TEORICO - 1:+.2%}), separates {acierto_c:.2%} of the pairs']

    texto = '\n'.join(lineas)
    print('\n' + texto)
    ruta = os.path.join(args.diagnosticos, f'{args.nombre}_resultados.txt')
    with open(ruta, 'w', encoding='utf-8') as archivo:
        archivo.write(texto + '\n')
    print(f'Saved {ruta}')
    np.savez(os.path.join(args.diagnosticos, f'{args.nombre}_pares.npz'), a=a_pares, Dv=Dv_pares,
             probabilidad=probabilidad, patron=patron, mediana_log_sigma=np.median(muestras_pares, axis=1))

    # Same plot as the simulations, with the theoretical threshold on top
    figura, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(a_pares[patron], Dv_pares[patron], s=2, color='green', label=f'pattern ({patron.sum()})')
    ax.scatter(a_pares[~patron], Dv_pares[~patron], s=2, color='black', label=f'no pattern ({(~patron).sum()})')
    a_curva = np.linspace(cfg['a_min'], cfg['a_max'], 200)
    ax.plot(a_curva, C_TEORICO / a_curva, 'r-', linewidth=2, label=r'$D_v = (3 + 2\sqrt{2})/a$')
    ax.set_ylim(0, cfg['dv_max'])
    ax.set_xlabel('a')
    ax.set_ylabel('Dv')
    ax.set_title(f'Network predictions on {args.pares} (a, Dv) pairs')
    ax.legend(loc='upper right', markerscale=4)
    guardar_figura(figura, 'patron')
