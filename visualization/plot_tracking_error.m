function plot_tracking_error(results_no_opt, results_opt, exp_id)
% Erreur de tracking position robot

figure('Position', [100, 100, 1200, 400]);

%% Erreur position (norme)
subplot(1,2,1);
hold on; grid on;

error_no_opt = vecnorm(results_no_opt.pos_error);
error_opt = vecnorm(results_opt.pos_error);

plot(results_no_opt.t, error_no_opt*1000, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Non-optimized');
plot(results_opt.t, error_opt*1000, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Optimized');

xlabel('Time [s]', 'FontSize', 11);
ylabel('Position Error [mm]', 'FontSize', 11);
title('CoM Tracking Error', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);

% RMS
rms_no_opt = rms(error_no_opt);
rms_opt = rms(error_opt);
text(0.6, 0.9, sprintf('RMS no-opt: %.2f mm', rms_no_opt*1000), ...
     'Units', 'normalized', 'FontSize', 9, 'Color', 'r');
text(0.6, 0.8, sprintf('RMS opt: %.2f mm', rms_opt*1000), ...
     'Units', 'normalized', 'FontSize', 9, 'Color', 'b');

%% Erreur par axe
subplot(1,2,2);
hold on; grid on;

plot(results_opt.t, results_opt.pos_error(1,:)*1000, 'r-', 'LineWidth', 1.2, 'DisplayName', 'e_x');
plot(results_opt.t, results_opt.pos_error(2,:)*1000, 'g-', 'LineWidth', 1.2, 'DisplayName', 'e_y');
plot(results_opt.t, results_opt.pos_error(3,:)*1000, 'b-', 'LineWidth', 1.2, 'DisplayName', 'e_z');

xlabel('Time [s]', 'FontSize', 11);
ylabel('Position Error [mm]', 'FontSize', 11);
title('Tracking Error per Axis (Optimized)', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_opt.t)]);
grid on;

%% Titre global
sgtitle(sprintf('Experiment #%d: Tracking Performance', exp_id), 'FontSize', 14, 'FontWeight', 'bold');

%% Sauvegarde
saveas(gcf, sprintf('results/figures/exp%d_tracking.png', exp_id));
fprintf('  ✓ Tracking plot saved\n');

end