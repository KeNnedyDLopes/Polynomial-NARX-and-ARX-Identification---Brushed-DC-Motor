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
import os

## Código

## Diretório contendo os arquivos do banco de dados
os.chdir(r"C:\Users\kened\OneDrive\UFSJ\TCC\Datasets\DC\CSV_Novos")


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
    # Cálculo do RRSE
    rrse = root_relative_squared_error(y_valid, yhat)
    # Cálculo do RMSE (Erro em RPM)
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
    ax.axis('off') # Esconde os eixos matemáticos
    
    cabecalho = f"Modelagem: {titulo}\nRRSE: {rrse:.5f}  |  RMSE: {rmse:.2f} RPM"
    ax.set_title(cabecalho, fontsize=12, fontweight='bold', color='#2E86C1', pad=20)
    
    # Renderiza o DataFrame como uma tabela no matplotlib
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
    
    # Plota apenas o número de pontos definido em n_plot
    plt.plot(y_valid[:n_plot], label='Data', marker='o', linestyle='-', linewidth=1.5)
    plt.plot(yhat[:n_plot], label='Model', marker='*', linestyle='-', linewidth=1.5)
    
    # Aplica o título dinâmico
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


## EXECUÇÃO ----------------------------------------

# CARREGAMENTO DE TODOS OS DADOS

# 1. Dinâmica Rápida (PRBS)
x_treino_p9, y_treino_p9 = carregar_dados('PRBS9_Base.csv')
x_teste_p9, y_teste_p9   = carregar_dados('PRBS9_Teste.csv')

# 2. Generalização (Motor A Completo)
x_prbs9_B, y_prbs9_B = carregar_dados('PRBS9_Completo_MotorA.csv')
x_seno_B, y_seno_B   = carregar_dados('Seno10Hz_MotorA.csv', corte_inicial=50)
#x_tri_B, y_tri_B     = carregar_dados('Tri10Hz_MotorA.csv', corte_inicial=50)
#x_quad_B, y_quad_B   = carregar_dados('Quad10Hz_MotorA.csv')
#x_s10_B, y_s10_B     = carregar_dados('Step_noLoad_10_MotorA.csv')
x_s60_B, y_s60_B     = carregar_dados('Step_noLoad_60_MotorA.csv')
#x_s100_B, y_s100_B   = carregar_dados('Step_noLoad_100_MotorA.csv')
#x_sl100_B, y_sl100_B = carregar_dados('Step_Load_100_MotorA.csv')


print("Todos os arquivos foram carregados com sucesso!")


# ------------------------------------------------------------------------------
# 1. MODELOS PRBS 9 BITS
# ------------------------------------------------------------------------------

# Chamada do modelo ARX (Grau 1)
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

# Resultados da predição (Treino)
#exibir_resultados_modelo(
#    titulo="ARX - Dinâmica Rápida (PRBS 9 - Ajuste PWM)",
#    df_equacao=eq_arx_p9, rrse=rrse_arx_p9, rmse=rmse_arx_p9,
#    y_valid=y_teste_p9, yhat=yhat_arx_p9, n_plot=600
#)

# ------------------------------------------------------------------------------
# Chamada do modelo NARX (Grau 2)
# ------------------------------------------------------------------------------
modelo_narx_p9, yhat_narx_p9, rrse_narx_p9, rmse_narx_p9, eq_narx_p9 = modelo_narx(
    x_train=x_treino_p9,
    y_train=y_treino_p9,
    x_valid=x_teste_p9,
    y_valid=y_teste_p9,
    grau=2,
    ylag=2,               
    xlag=2,              
    selecao_ordem=True,
    n_info=6,
    n_termos_fixo=6
)

# Resultados da predição (Treino)
# exibir_resultados_modelo(
#     titulo="NARX - Dinâmica Rápida (PRBS 9 - Ajuste Definitivo PWM)",
#     df_equacao=eq_narx_p9, rrse=rrse_narx_p9, rmse=rmse_narx_p9,
#     y_valid=y_teste_p9, yhat=yhat_narx_p9, n_plot=600
# )

# TESTE DE GENERALIZAÇÃO

# ------------------------------------------------------------------------------
# 1. GENERALIZAÇÃO PRBS 9 BITS
# ------------------------------------------------------------------------------

# Previsão usando o ARX P9 já treinado
yhat_arx_p9_B = modelo_arx_p9.predict(X=x_prbs9_B, y=y_prbs9_B, steps_ahead=1)
rrse_arx_p9_B = root_relative_squared_error(y_prbs9_B, yhat_arx_p9_B)
rmse_arx_p9_B = np.sqrt(np.mean((y_prbs9_B - yhat_arx_p9_B)**2))

# Previsão usando o NARX P9 já treinado
yhat_narx_p9_B = modelo_narx_p9.predict(X=x_prbs9_B, y=y_prbs9_B, steps_ahead=1)
rrse_narx_p9_B = root_relative_squared_error(y_prbs9_B, yhat_narx_p9_B)
rmse_narx_p9_B = np.sqrt(np.mean((y_prbs9_B - yhat_narx_p9_B)**2))


# Resultados da predição (Validação)
#exibir_resultados_modelo(
#    titulo="ARX - Treinamento (PRBS 9)",
#    df_equacao=eq_arx_p9, rrse=rrse_arx_p9, rmse=rmse_arx_p9,
#    y_valid=y_teste_p9, yhat=yhat_arx_p9, n_plot=600
#)
#
# Resultados da predição (Validação)
#exibir_resultados_modelo(
#    titulo="NARX - Treinamento (PRBS 9)",
#    df_equacao=eq_narx_p9, rrse=rrse_narx_p9, rmse=rmse_narx_p9,
#    y_valid=y_teste_p9, yhat=yhat_narx_p9, n_plot=600
#)

# 2. RESULTADOS DA GENERALIZAÇÃO
exibir_resultados_modelo(
    titulo="ARX vs PRBS9 (Motor A)",
    df_equacao=eq_arx_p9, rrse=rrse_arx_p9_B, rmse=rmse_arx_p9_B,
    y_valid=y_prbs9_B, yhat=yhat_arx_p9_B, n_plot=2000
)

exibir_resultados_modelo(
    titulo="NARX vs PRBS9 (Motor A)",
    df_equacao=eq_narx_p9, rrse=rrse_narx_p9_B, rmse=rmse_narx_p9_B,
    y_valid=y_prbs9_B, yhat=yhat_narx_p9_B, n_plot=2000
)

# 3. VALIDAÇÃO CRUZADA EM OUTRAS DINÂMICAS (SENOIDE E STEP)
cenarios_validacao = [
    ("Senoide 10Hz", x_seno_B, y_seno_B, 300),
    ("Step", x_s60_B, y_s60_B, 400),
]

for nome_sinal, x_val, y_val, n_pontos in cenarios_validacao:
    
    # Previsão e Exibição ARX
    yhat_arx = modelo_arx_p9.predict(X=x_val, y=y_val, steps_ahead=1)
    rrse_arx = root_relative_squared_error(y_val, yhat_arx)
    rmse_arx = np.sqrt(np.mean((y_val - yhat_arx)**2))
    
    exibir_resultados_modelo(
        titulo=f"ARX P9 Prevendo: {nome_sinal}",
        df_equacao=eq_arx_p9, rrse=rrse_arx, rmse=rmse_arx,
        y_valid=y_val, yhat=yhat_arx, n_plot=n_pontos
    )
    
    # Previsão e Exibição NARX
    yhat_narx = modelo_narx_p9.predict(X=x_val, y=y_val, steps_ahead=1)
    rrse_narx = root_relative_squared_error(y_val, yhat_narx)
    rmse_narx = np.sqrt(np.mean((y_val - yhat_narx)**2))
    
    exibir_resultados_modelo(
        titulo=f"NARX P9 Prevendo: {nome_sinal}",
        df_equacao=eq_narx_p9, rrse=rrse_narx, rmse=rmse_narx,
        y_valid=y_val, yhat=yhat_narx, n_plot=n_pontos
    )