function plot_satellite_rotation(results_no_opt, results_opt, exp_id, env)
% Reproduit Figure 6, 9, 12 du papier Lutze

figure('Position', [100, 100, 1200, 400]);

%% Subplot 1: Alpha
subplot(1,3,1);
hold on; grid on;
plot(results_no_opt.t, rad2deg(results_no_opt.alpha), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Non-optimized');
plot(results_opt.t, rad2deg(results_opt.alpha), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Optimized');
xlabel('Time [s]', 'FontSize', 11);
ylabel('\alpha [deg]', 'FontSize', 11);
title('Roll Angle', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);

%% Subplot 2: Beta (angle critique)
subplot(1,3,2);
hold on; grid on;
plot(results_no_opt.t, rad2deg(results_no_opt.beta), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Non-optimized');
plot(results_opt.t, rad2deg(results_opt.beta), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Optimized');
xlabel('Time [s]', 'FontSize', 11);
ylabel('\beta [deg]', 'FontSize', 11);
title('Pitch Angle (Critical)', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);

% Annotation max beta
beta_max_no_opt = max(abs(results_no_opt.beta));
beta_max_opt = max(abs(results_opt.beta));
text(0.5, 0.95, sprintf('|\\beta|_{max} no-opt: %.3f°', rad2deg(beta_max_no_opt)), ...
     'Units', 'normalized', 'FontSize', 9, 'Color', 'r');
text(0.5, 0.85, sprintf('|\\beta|_{max} opt: %.4f°', rad2deg(beta_max_opt)), ...
     'Units', 'normalized', 'FontSize', 9, 'Color', 'b');

%% Subplot 3: Gamma
subplot(1,3,3);
hold on; grid on;
plot(results_no_opt.t, rad2deg(results_no_opt.gamma), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Non-optimized');
plot(results_opt.t, rad2deg(results_opt.gamma), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Optimized');
xlabel('Time [s]', 'FontSize', 11);
ylabel('\gamma [deg]', 'FontSize', 11);
title('Yaw Angle', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);

%% Titre global
sgtitle(sprintf('Experiment #%d: Satellite Attitude Angles', exp_id), 'FontSize', 14, 'FontWeight', 'bold');

%% Sauvegarde
saveas(gcf, sprintf('results/figures/exp%d_rotation.png', exp_id));
fprintf('  ✓ Rotation plot saved\n');

end