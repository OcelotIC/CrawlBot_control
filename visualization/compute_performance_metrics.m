function compute_performance_metrics(results_no_opt, results_opt, env)
% Affiche métriques quantitatives

fprintf('┌─────────────────────────────────────────────────────────┐\n');
fprintf('│ METRIC                    │ NO-OPT    │ OPTIMIZED     │\n');
fprintf('├─────────────────────────────────────────────────────────┤\n');

%% Rotation satellite
alpha_max_no = rad2deg(max(abs(results_no_opt.alpha)));
alpha_max_opt = rad2deg(max(abs(results_opt.alpha)));
fprintf('│ Max |α| [deg]             │ %8.4f  │ %8.4f      │\n', alpha_max_no, alpha_max_opt);

beta_max_no = rad2deg(max(abs(results_no_opt.beta)));
beta_max_opt = rad2deg(max(abs(results_opt.beta)));
fprintf('│ Max |β| [deg]             │ %8.4f  │ %8.4f      │\n', beta_max_no, beta_max_opt);

gamma_max_no = rad2deg(max(abs(results_no_opt.gamma)));
gamma_max_opt = rad2deg(max(abs(results_opt.gamma)));
fprintf('│ Max |γ| [deg]             │ %8.4f  │ %8.4f      │\n', gamma_max_no, gamma_max_opt);

fprintf('├─────────────────────────────────────────────────────────┤\n');

%% Forces contact
F_rms_no = rms(vecnorm(results_no_opt.Fc(1:3,:)));
F_rms_opt = rms(vecnorm(results_opt.Fc(1:3,:)));
fprintf('│ RMS ||F_c|| [N]           │ %8.1f  │ %8.1f      │\n', F_rms_no, F_rms_opt);

F_max_no = max(vecnorm(results_no_opt.Fc(1:3,:)));
F_max_opt = max(vecnorm(results_opt.Fc(1:3,:)));
fprintf('│ Max ||F_c|| [N]           │ %8.1f  │ %8.1f      │\n', F_max_no, F_max_opt);

fprintf('├─────────────────────────────────────────────────────────┤\n');

%% Tracking
err_rms_no = rms(vecnorm(results_no_opt.pos_error)) * 1000;
err_rms_opt = rms(vecnorm(results_opt.pos_error)) * 1000;
fprintf('│ RMS tracking error [mm]   │ %8.2f  │ %8.2f      │\n', err_rms_no, err_rms_opt);

err_max_no = max(vecnorm(results_no_opt.pos_error)) * 1000;
err_max_opt = max(vecnorm(results_opt.pos_error)) * 1000;
fprintf('│ Max tracking error [mm]   │ %8.2f  │ %8.2f      │\n', err_max_no, err_max_opt);

fprintf('├─────────────────────────────────────────────────────────┤\n');

%%  NOUVEAU: Saturation RW
h_max = env.RW.h_total_max;
sat_max_no = max(vecnorm(results_no_opt.h_RW)) / h_max * 100;
sat_max_opt = max(vecnorm(results_opt.h_RW)) / h_max * 100;
fprintf('│  Max RW saturation [%%]  │ %8.1f  │ %8.1f      │\n', sat_max_no, sat_max_opt);

sat_avg_no = mean(vecnorm(results_no_opt.h_RW)) / h_max * 100;
sat_avg_opt = mean(vecnorm(results_opt.h_RW)) / h_max * 100;
fprintf('│  Avg RW saturation [%%]  │ %8.1f  │ %8.1f      │\n', sat_avg_no, sat_avg_opt);

fprintf('└─────────────────────────────────────────────────────────┘\n');

%% Warnings
if sat_max_no > 90 || sat_max_opt > 90
    fprintf('\n️  WARNING: Critical RW saturation detected!\n');
    fprintf('   → Desaturation maneuver required\n');
    fprintf('   → Consider NMPC with explicit momentum constraints\n\n');
end

%% Improvement ratio
improvement_beta = (beta_max_no - beta_max_opt) / beta_max_no * 100;
fprintf('\nℹ️  β improvement: %.1f%%\n', improvement_beta);

if improvement_beta < 50
    fprintf('⚠️  Limited improvement suggests trajectory crosses satellite CoM\n');
end

end