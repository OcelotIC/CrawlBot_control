function plot_contact_wrenches(results_no_opt, results_opt, exp_id)
% Plot forces et moments de contact

figure('Position', [100, 100, 1200, 600]);

%% Forces
subplot(2,1,1);
hold on; grid on;

% Non-optimized
plot(results_no_opt.t, vecnorm(results_no_opt.Fc(1:3,:)), 'r--', 'LineWidth', 1.5, 'DisplayName', 'No-opt ||F||');

% Optimized
plot(results_opt.t, vecnorm(results_opt.Fc(1:3,:)), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Opt ||F||');

xlabel('Time [s]', 'FontSize', 11);
ylabel('Contact Force [N]', 'FontSize', 11);
title('Contact Force Magnitude', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);

%% Moments
subplot(2,1,2);
hold on; grid on;

% Non-optimized
plot(results_no_opt.t, vecnorm(results_no_opt.Fc(4:6,:)), 'r--', 'LineWidth', 1.5, 'DisplayName', 'No-opt ||\tau||');

% Optimized
plot(results_opt.t, vecnorm(results_opt.Fc(4:6,:)), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Opt ||\tau||');

xlabel('Time [s]', 'FontSize', 11);
ylabel('Contact Moment [Nm]', 'FontSize', 11);
title('Contact Moment Magnitude', 'FontSize', 12);
legend('Location', 'best');
xlim([0, max(results_no_opt.t)]);

%% Titre global
sgtitle(sprintf('Experiment #%d: Contact Wrenches', exp_id), 'FontSize', 14, 'FontWeight', 'bold');

%% Sauvegarde
saveas(gcf, sprintf('results/figures/exp%d_wrenches.png', exp_id));
fprintf('  ✓ Wrenches plot saved\n');

end