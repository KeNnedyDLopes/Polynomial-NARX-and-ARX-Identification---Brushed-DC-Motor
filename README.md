# Polynomial NARX and ARX Identification - Brushed DC Motor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![SysIdentPy](https://img.shields.io/badge/Library-SysIdentPy-orange)
![MATLAB](https://img.shields.io/badge/Plots-MATLAB-0076A8)
![UFSJ](https://img.shields.io/badge/UFSJ-Engenharia%20El%C3%A9trica-success)

Repositório com as rotinas computacionais desenvolvidas para o Trabalho de Conclusão de Curso (TCC) em **Engenharia Elétrica** pela **Universidade Federal de São João del-Rei (UFSJ)**.

* **Autor:** Kennedy David Lopes Alves
* **Orientador:** Prof. Dr. Samir Angelo Milani Martins

---

## Sobre o Trabalho

Este projeto realiza a identificação caixa-preta de um atuador eletromecânico composto por um **motor de corrente contínua com escovas acoplado a uma caixa de redução por rosca sem-fim (*worm gear*)**, operado a uma taxa de amostragem de $100\text{ Hz}$.

O objetivo é comparar o desempenho preditivo entre uma estrutura linear (**ARX**) e uma não-linear polinomial (**NARX**), selecionadas via algoritmo **FROLS** (*Forward Regression Orthogonal Least Squares*) e avaliadas em regime de simulação livre (**Free Run**).

### Metodologia e Ensaios
* **Treinamento:** Sinal pseudoaleatório **PRBS de 9 bits**.
* **Validação e Generalização:** Resposta ao **PRBS de 9 bits**, **onda senoidal ($10\text{ Hz}$)** e **degrau de tensão (*Step*)**.
* **Critérios de Avaliação:** Métricas globais de erro (**RMSE** e **RRSE**) e análise estatística de **autocorrelação dos resíduos**.

---

## Arquivos Disponíveis

* `identificacao_arx_narx.py` — Pré-processamento dos sinais, estimação dos modelos ARX e NARX via `SysIdentPy`, simulação em *Free Run* e exportação dos resultados para `.mat`.
* `gerar_graficos_tcc.m` — Rotina em MATLAB para geração das figuras vetoriais de rastreio dinâmico e autocorrelação dos resíduos.

---

## Como Executar

1. Instale as dependências necessárias em Python:
   ```bash
   pip install sysidentpy numpy pandas scipy matplotlib
