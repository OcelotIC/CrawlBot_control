function plot_momentum_saturation(results_no_opt, results_opt, exp_id, env)
% ⭐ NOUVELLE ANALYSE: Saturation reaction wheels
% NON PRÉSENTE DANS LE PAPIER ORIGINAL

figure('Position', [100, 100, 1200, 600]);

%% Calcul capacité restante RW
h_max = env.RW.h_total_max;

h_norm_no_opt = vecnorm(results_no_opt.h_RW);
h_norm_opt = vecnorm(results_opt.h_RW);

saturation_no_opt = (h_norm_no_opt / h_max) * 100;
saturation_opt = (h_norm_opt / h_max) * 100;

%% Subplot 1: Moment angulaire RW
subplot(2,1,1);
hold on; grid on;

plot(results_no_opt.t, h_norm_no_opt, 'r--', 'LineWidth', 1.5, 'DisplayName', 'No-opt ||h_{RW}||');
plot(results_opt.t, h_norm_opt, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Opt ||h_{RW}||');
yline(h_max, 'k--', 'LineWidth', 2, 'DisplayName', 'h_{max}');

xlabel('Time [s]', 'FontSize', 11);
ylabel('Angular Momentum [Nms]', 'FontSize', 11);
title('Reaction Wheel Momentum Storage', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);
ylim([0, h_max*1.1]);

% Warning zone
fill([0 max(results_no_opt.t) max(results_no_opt.t) 0], ...
     [h_max*0.9 h_max*0.9 h_max*1.1 h_max*1.1], ...
     'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

%% Subplot 2: Saturation en %
subplot(2,1,2);
hold on; grid on;

plot(results_no_opt.t, saturation_no_opt, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Non-optimized');
plot(results_opt.t, saturation_opt, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Optimized');
yline(100, 'k--', 'LineWidth', 2, 'DisplayName', 'Saturation');
yline(90, 'r:', 'LineWidth', 1.5, 'DisplayName', 'Critical (90%)');

xlabel('Time [s]', 'FontSize', 11);
ylabel('Saturation [%]', 'FontSize', 11);
title('RW Saturation Level', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);
ylim([0, 110]);

% Warning zone
fill([0 max(results_no_opt.t) max(results_no_opt.t) 0], ...
     [90 90 110 110], ...
     'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

% Max saturation annotation
sat_max_no_opt = max(saturation_no_opt);
sat_max_opt = max(saturation_opt);
text(0.05, 0.9, sprintf('Max sat. no-opt: %.1f%%', sat_max_no_opt), ...
     'Units', 'normalized', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');
text(0.05, 0.8, sprintf('Max sat. opt: %.1f%%', sat_max_opt), ...
     'Units', 'normalized', 'FontSize', 10, 'Color', 'b', 'FontWeight', 'bold');

if sat_max_no_opt > 90 || sat_max_opt > 90
    text(0.05, 0.7, '⚠ CRITICAL SATURATION!', ...
         'Units', 'normalized', 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');
end

%% Titre global
sgtitle(sprintf('Experiment #%d: ⭐ NEW - Reaction Wheel Saturation Analysis', exp_id), ...
        'FontSize', 14, 'FontWeight', 'bold');

%% Sauvegarde
saveas(gcf, sprintf('results/figures/exp%d_momentum_saturation.png', exp_id));
fprintf('  ✓ Momentum saturation plot saved\n');

end