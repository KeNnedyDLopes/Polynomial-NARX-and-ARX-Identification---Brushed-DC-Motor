## Código para a Identificação de Sistemas utilizando Modelos ARX e NARX
## Elaborado por Kennedy David Lopes - 29/09/26

# Import das bibliotecas
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

## Diretório contendo os arquivos do banco de dados
os.chdir(r"C:\Users\kened\OneDrive\UFSJ\TCC\Datasets\DC\CSV_Novos")

# Pasta de saída para os arquivos .mat
pasta_saida = "arquivos_mat_tcc"
os.makedirs(pasta_saida, exist_ok=True)


## Função Carregamento dos dados de treinamento
def carregar_dados(caminho_csv, corte_inicial=0):
    try:
        df = pd.read_csv(caminho_csv, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(caminho_csv)

    df.columns = df.columns.str.strip()

    x = pd.to_numeric(df['PWM'], errors='coerce').values.reshape(-1, 1)
    y = pd.to_numeric(df['Velocity'], errors='coerce').values.reshape(-1, 1)

    x = np.nan_to_num(x.astype(np.float64), nan=0.0)
    y = np.nan_to_num(y.astype(np.float64), nan=0.0)

    # Normalização do PWM
    x = x / 255.0

    if corte_inicial > 0:
        x = x[corte_inicial:]
        y = y[corte_inicial:]

    return x, y


## Função para a geração de modelos ARX e NARX
def modelo_narx(x_train, y_train, x_valid, y_valid,
                grau, ylag, xlag,
                selecao_ordem, n_info, n_termos_fixo):

    # 1. Configuração Matemática Base
    basis_function = Polynomial(degree=grau)
    estimator = LeastSquares()

    # 2. Instanciação do algoritmo de seleção
    if selecao_ordem:
        model = FROLS(
            order_selection=True,
            info_criteria="aic",
            n_info_values=n_info,
            ylag=ylag,
            xlag=xlag,
            estimator=estimator,
            basis_function=basis_function,
            err_tol=None
        )
    else:
        # Se for falso força o modelo a não usar o AIC
        model = FROLS(
            order_selection=False,
            n_terms=n_termos_fixo,
            ylag=ylag,
            xlag=xlag,
            estimator=estimator,
            basis_function=basis_function
        )

    # 3. Treinamento
    model.fit(X=x_train, y=y_train)

    # 4. Validação (Previsão One Step Ahead) e Cálculos de Erro
    yhat = model.predict(X=x_valid, y=y_valid, steps_ahead=1)
    rrse = root_relative_squared_error(y_valid, yhat)
    rmse = np.sqrt(np.mean((y_valid - yhat)**2))

    # 5. Extração das Equações
    df_equacao = pd.DataFrame(
        results(
            model.final_model, model.theta, model.err,
            model.n_terms, err_precision=8, dtype='sci'
        ),
        columns=['Regressores', 'Pesos', 'ERR']
    )

    return model, yhat, rrse, rmse, df_equacao


## Função para exibir os resultados
def exibir_resultados_modelo(titulo, df_equacao, rrse, rmse, y_valid, yhat, n_plot=150):
    
    # 1. Criação do Gráfico de Texto (Métricas e Equação)
    fig, ax = plt.subplots(figsize=(10, max(3, len(df_equacao) * 0.5)))
    ax.axis('off')
    
    cabecalho = f"Modelagem: {titulo}\nRRSE: {rrse:.5f}  |  RMSE: {rmse:.2f} RPM"
    ax.set_title(cabecalho, fontsize=12, fontweight='bold', color='#2E86C1', pad=20)
    
    tabela = ax.table(cellText=df_equacao.values, 
                      colLabels=df_equacao.columns, 
                      loc='center', cellLoc='center')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1, 1.5)
    plt.tight_layout()
    plt.show()

    # 2. Gráfico de Rastreio Dinâmico (Real vs Previsto)
    plt.figure(figsize=(14, 3))
    plt.plot(y_valid[:n_plot], label='Data', marker='o', linestyle='-', linewidth=1.5)
    plt.plot(yhat[:n_plot], label='Model', marker='*', linestyle='-', linewidth=1.5)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('y, $\hat{y}$', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # 3. Gráfico de Autocorrelação dos Resíduos
    residuos_acf = compute_residues_autocorrelation(y_valid, yhat)
    plot_residues_correlation(data=residuos_acf, title=f"Resíduos- {titulo}", ylabel="Correlação")
    plt.show()


## Funções de exportação para MATLAB (.mat)
def exportar_predicao_mat(nome_arquivo, y_real, y_pred, Ts=0.01):
    """Exporta os vetores de comparação temporal (Real vs. Modelo) para .mat."""
    y_r = np.asarray(y_real, dtype=float).reshape(-1, 1)
    y_p = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    n_amostras = len(y_r)
    
    amostras = np.arange(n_amostras, dtype=float).reshape(-1, 1)
    tempo = amostras * Ts
    
    caminho = os.path.join(pasta_saida, nome_arquivo)
    sio.savemat(caminho, {
        "y_real": y_r,
        "y_modelo": y_p,
        "amostras": amostras,
        "tempo": tempo,
        "Ts": float(Ts)
    })


def exportar_residuos_mat(nome_arquivo, y_real, y_pred, max_lag=100):
    """Calcula e exporta a autocorrelação dos resíduos idêntica ao SysIdentPy."""
    y_r = np.asarray(y_real, dtype=float).reshape(-1, 1)
    y_p = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    
    # Usa a função nativa do SysIdentPy (preserva o offset médio do resíduo)
    acf, limite_sup, limite_inf = compute_residues_autocorrelation(y_r, y_p)
    
    acf = np.asarray(acf, dtype=float).flatten()[:max_lag]
    lags = np.arange(len(acf), dtype=float).reshape(-1, 1)
    residuo = (y_r - y_p).reshape(-1, 1)
    
    caminho = os.path.join(pasta_saida, nome_arquivo)
    sio.savemat(caminho, {
        "residuo": residuo,
        "acf": acf.reshape(-1, 1),
        "lags": lags,
        "limite_superior": float(limite_sup),
        "limite_inferior": float(limite_inf)
    })


## EXECUÇÃO ----------------------------------------

# 1. Dinâmica Rápida (PRBS)
x_treino_p9, y_treino_p9 = carregar_dados('PRBS9_Base.csv')
x_teste_p9, y_teste_p9   = carregar_dados('PRBS9_Teste.csv')

# 2. Generalização (Motor A Completo)
x_prbs9_B, y_prbs9_B = carregar_dados('PRBS9_Completo_MotorA.csv')
x_seno_B, y_seno_B   = carregar_dados('Seno10Hz_MotorA.csv', corte_inicial=50)
x_s60_B, y_s60_B     = carregar_dados('Step_noLoad_60_MotorA.csv')

print("Todos os arquivos foram carregados com sucesso!")

# ------------------------------------------------------------------------------
# 1. TREINAMENTO DOS MODELOS PRBS 9 BITS
# ------------------------------------------------------------------------------
modelo_arx_p9, yhat_arx_p9, rrse_arx_p9, rmse_arx_p9, eq_arx_p9 = modelo_narx(
    x_train=x_treino_p9,
    y_train=y_treino_p9,
    x_valid=x_teste_p9,
    y_valid=y_teste_p9,
    grau=1,
    ylag=2,  
    xlag=2,
    selecao_ordem=True, 
    n_info=5,
    n_termos_fixo=5  
)

modelo_narx_p9, yhat_narx_p9, rrse_narx_p9, rmse_narx_p9, eq_narx_p9 = modelo_narx(
    x_train=x_treino_p9,
    y_train=y_treino_p9,
    x_valid=x_teste_p9,
    y_valid=y_teste_p9,
    grau=2,
    ylag=2,                
    xlag=1,              

    selecao_ordem=True,
    n_info=7,
    n_termos_fixo=7
)

# ------------------------------------------------------------------------------
# 2. VALIDAÇÃO PRBS 9 BITS (Motor A) E EXPORTAÇÃO .MAT
# ------------------------------------------------------------------------------
yhat_arx_p9_B = modelo_arx_p9.predict(X=x_prbs9_B, y=y_prbs9_B)
rrse_arx_p9_B = root_relative_squared_error(y_prbs9_B, yhat_arx_p9_B)
rmse_arx_p9_B = np.sqrt(np.mean((y_prbs9_B - yhat_arx_p9_B)**2))

yhat_narx_p9_B = modelo_narx_p9.predict(X=x_prbs9_B, y=y_prbs9_B)
rrse_narx_p9_B = root_relative_squared_error(y_prbs9_B, yhat_narx_p9_B)
rmse_narx_p9_B = np.sqrt(np.mean((y_prbs9_B - yhat_narx_p9_B)**2))

exibir_resultados_modelo(
    titulo="ARX vs PRBS9 (Motor A)",
    df_equacao=eq_arx_p9, rrse=rrse_arx_p9_B, rmse=rmse_arx_p9_B,
    y_valid=y_prbs9_B, yhat=yhat_arx_p9_B, n_plot=2000
)
exportar_predicao_mat("prbs9_arx.mat", y_prbs9_B, yhat_arx_p9_B)
exportar_residuos_mat("prbs9_residuos_arx.mat", y_prbs9_B, yhat_arx_p9_B)

exibir_resultados_modelo(
    titulo="NARX vs PRBS9 (Motor A)",
    df_equacao=eq_narx_p9, rrse=rrse_narx_p9_B, rmse=rmse_narx_p9_B,
    y_valid=y_prbs9_B, yhat=yhat_narx_p9_B, n_plot=2000
)
exportar_predicao_mat("prbs9_narx.mat", y_prbs9_B, yhat_narx_p9_B)
exportar_residuos_mat("prbs9_residuos_narx.mat", y_prbs9_B, yhat_narx_p9_B)

# ------------------------------------------------------------------------------
# 3. VALIDAÇÃO CRUZADA (SENOIDE E STEP) E EXPORTAÇÃO .MAT
# ------------------------------------------------------------------------------
cenarios_validacao = [
    ("Senoide 10Hz", "senoide", x_seno_B, y_seno_B, 300),
    ("Step", "step", x_s60_B, y_s60_B, 400),
]

for nome_sinal, prefixo, x_val, y_val, n_pontos in cenarios_validacao:
    
    # Previsão, Exibição e Exportação ARX
    yhat_arx = modelo_arx_p9.predict(X=x_val, y=y_val)
    rrse_arx = root_relative_squared_error(y_val, yhat_arx)
    rmse_arx = np.sqrt(np.mean((y_val - yhat_arx)**2))
    
    exibir_resultados_modelo(
        titulo=f"ARX P9 Prevendo: {nome_sinal}",
        df_equacao=eq_arx_p9, rrse=rrse_arx, rmse=rmse_arx,
        y_valid=y_val, yhat=yhat_arx, n_plot=n_pontos
    )
    exportar_predicao_mat(f"{prefixo}_arx.mat", y_val, yhat_arx)
    exportar_residuos_mat(f"{prefixo}_residuos_arx.mat", y_val, yhat_arx)
    
    # Previsão, Exibição e Exportação NARX
    yhat_narx = modelo_narx_p9.predict(X=x_val, y=y_val)
    rrse_narx = root_relative_squared_error(y_val, yhat_narx)
    rmse_narx = np.sqrt(np.mean((y_val - yhat_narx)**2))
    
    exibir_resultados_modelo(
        titulo=f"NARX P9 Prevendo: {nome_sinal}",
        df_equacao=eq_narx_p9, rrse=rrse_narx, rmse=rmse_narx,
        y_valid=y_val, yhat=yhat_narx, n_plot=n_pontos
    )
    exportar_predicao_mat(f"{prefixo}_narx.mat", y_val, yhat_narx)
    exportar_residuos_mat(f"{prefixo}_residuos_narx.mat", y_val, yhat_narx)

print(f"12 arquivos .mat exportados com sucesso em: {os.path.abspath(pasta_saida)}")