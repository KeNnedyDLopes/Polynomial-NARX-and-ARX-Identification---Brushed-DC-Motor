%% Script para Geração dos Gráficos de Predição e Resíduos
clear; clc; close all;

% Diretório onde os 12 arquivos .mat foram salvos pelo Python
pasta_mat = 'C:\Users\kened\OneDrive\UFSJ\TCC\Datasets\DC\CSV_Novos\arquivos_mat_tcc';

% Pasta de saída onde as figuras prontas (PDF vetorial e PNG 300 DPI) serão salvas
pasta_figuras = fullfile(pasta_mat, 'figuras_tcc');
if ~exist(pasta_figuras, 'dir')
    mkdir(pasta_figuras);
end

% Definição das cores RGB normalizadas [0 a 1]
cor_azul     = [0, 0, 1];             % RGB (0, 0, 255) - Dados experimentais e ACF
cor_vermelho = [1, 0, 0];             % RGB (255, 0, 0) - Modelo de predição
cor_faixa    = [0.85, 0.90, 1.00];    % Faixa de confiança 95%

% Configuração tipográfica padrão
fonte_padrao = 'Times New Roman';
tam_titulo   = 20;
tam_eixos    = 16;

%% ========================================================================
%  1. GRÁFICOS DE RASTREIO DINÂMICO (REAL VS. MODELO) - 6 FIGURAS
%  ========================================================================
% Estrutura: {arquivo_mat, titulo_grafico, qtd_amostras_plot, nome_arquivo_saida}
configs_predicao = {
    'prbs9_arx.mat',    'Modelo ARX prevendo o sinal PRBS de 9 Bits',   600, 'arx_prbs9';
    'prbs9_narx.mat',   'Modelo NARX prevendo o sinal PRBS de 9 Bits',  600, 'narx_prbs9';
    'senoide_arx.mat',  'Modelo ARX prevendo o sinal Senóide',          300, 'arx_senoide10hz';
    'senoide_narx.mat', 'Modelo NARX prevendo o sinal Senóide',         300, 'narx_senoide10hz';
    'step_arx.mat',     'Modelo ARX prevendo o sinal Step',             250, 'arx_step';
    'step_narx.mat',    'Modelo NARX prevendo o sinal Step',            250, 'narx_step'
};

for i = 1:size(configs_predicao, 1)
    arq_mat  = configs_predicao{i, 1};
    titulo   = configs_predicao{i, 2};
    n_plot   = configs_predicao{i, 3};
    nome_out = configs_predicao{i, 4};
    
    dados = load(fullfile(pasta_mat, arq_mat));
    
    limite = min(n_plot, length(dados.y_real));
    idx = 1:limite;
    
    % Janela dimensionada para acomodar fonte 16/20 sem cortes
    fig = figure('Color', 'w', 'Position', [100, 100, 1150, 340]);
    
    plot(dados.amostras(idx), dados.y_real(idx), '-o', ...
        'Color', cor_azul, 'MarkerFaceColor', cor_azul, ...
        'MarkerSize', 3.5, 'LineWidth', 1.3, 'DisplayName', 'Data');
    hold on;
    plot(dados.amostras(idx), dados.y_modelo(idx), '-p', ...
        'Color', cor_vermelho, 'MarkerFaceColor', cor_vermelho, ...
        'MarkerSize', 4.0, 'LineWidth', 1.3, 'DisplayName', 'Model');
    
    ax = gca;
    set(ax, 'FontName', fonte_padrao, 'FontSize', tam_eixos, 'LineWidth', 0.9);
    
    title(titulo, 'FontName', fonte_padrao, 'FontSize', tam_titulo, 'FontWeight', 'bold');
    xlabel('Samples', 'FontName', fonte_padrao, 'FontSize', tam_eixos);
    ylabel('y, y_{hat}', 'FontName', fonte_padrao, 'FontSize', tam_eixos);
    
    % Legenda posicionada no canto inferior direito ('southeast')
    leg = legend('Location', 'southeast');
    set(leg, 'FontName', fonte_padrao, 'FontSize', tam_eixos);
    
    xlim([0, limite]);
    box on;
    
    % Exporta em PDF vetorial e PNG 300 DPI
    exportgraphics(fig, fullfile(pasta_figuras, [nome_out '.pdf']), 'ContentType', 'vector');
    exportgraphics(fig, fullfile(pasta_figuras, [nome_out '.png']), 'Resolution', 300);
end

%% ========================================================================
%  2. GRÁFICOS DE AUTOCORRELAÇÃO DOS RESÍDUOS - 6 FIGURAS
%  ========================================================================
% Estrutura: {arquivo_mat, titulo_grafico, nome_arquivo_saida}
configs_residuos = {
    'prbs9_residuos_arx.mat',    'Resíduos - Modelo ARX prevendo o sinal PRBS de 9 Bits',   'residuos_arx_prbs9';
    'prbs9_residuos_narx.mat',   'Resíduos - Modelo NARX prevendo o sinal PRBS de 9 Bits',  'residuos_narx_prbs9';
    'senoide_residuos_arx.mat',  'Resíduos - Modelo ARX prevendo o sinal Senóide',          'residuos_arx_senoide10hz';
    'senoide_residuos_narx.mat', 'Resíduos - Modelo NARX prevendo o sinal Senóide',         'residuos_narx_senoide10hz';
    'step_residuos_arx.mat',     'Resíduos - Modelo ARX prevendo o sinal Step',             'residuos_arx_step';
    'step_residuos_narx.mat',    'Resíduos - Modelo NARX prevendo o sinal Step',            'residuos_narx_step'
};

for i = 1:size(configs_residuos, 1)
    arq_mat  = configs_residuos{i, 1};
    titulo   = configs_residuos{i, 2};
    nome_out = configs_residuos{i, 3};
    
    dados = load(fullfile(pasta_mat, arq_mat));
    
    fig = figure('Color', 'w', 'Position', [150, 150, 860, 500]);
    
    x_min   = min(dados.lags);
    x_max   = max(dados.lags);
    lim_sup = dados.limite_superior;
    lim_inf = dados.limite_inferior;
    
    fill([x_min, x_max, x_max, x_min], [lim_inf, lim_inf, lim_sup, lim_sup], ...
        cor_faixa, 'EdgeColor', 'none');
    hold on;
    
    plot(dados.lags, dados.acf, '-', 'Color', cor_azul, 'LineWidth', 1.5);
    
    ax = gca;
    set(ax, 'FontName', fonte_padrao, 'FontSize', tam_eixos, 'LineWidth', 0.9, 'Layer', 'top');
    
    title(titulo, 'FontName', fonte_padrao, 'FontSize', tam_titulo, 'FontWeight', 'bold');
    xlabel('Lag', 'FontName', fonte_padrao, 'FontSize', tam_eixos);
    ylabel('Correlação', 'FontName', fonte_padrao, 'FontSize', tam_eixos);
    
    ylim([-1, 1]);
    xlim([x_min, x_max]);
    box on;
    
    % Exporta em PDF vetorial e PNG 300 DPI
    exportgraphics(fig, fullfile(pasta_figuras, [nome_out '.pdf']), 'ContentType', 'vector');
    exportgraphics(fig, fullfile(pasta_figuras, [nome_out '.png']), 'Resolution', 300);
end

fprintf('Todas as 12 figuras foram geradas e salvas em:\n%s\n', pasta_figuras);