classdef LutzeQPController < handle
    % Implémentation contrôleur QP Lutze et al. (2023)
    % Équations (12-22)
    % COMPATIBLE avec sys contenant cell arrays (versions _casadi)
    
    properties
        env
        Qr, Qb, Qc      % Matrices pondération
        Kr, Dr          % Gains robot PD
        Kb, Db          % Gains base PD
        F_min, F_max    % Limites SI
    end
    
    methods
        function obj = LutzeQPController(env)
            obj.env = env;
            obj.Qr = env.control.Qr;
            obj.Qb = env.control.Qb;
            obj.Qc = env.control.Qc;
            obj.Kr = env.control.Kr;
            obj.Dr = env.control.Dr;
            obj.Kb = env.control.Kb;
            obj.Db = env.control.Db;
            
            % Limites wrench (éq. 22)
            obj.F_min = [-env.SI.F_max*ones(3,1); -env.SI.tau_max*ones(3,1)];
            obj.F_max = [env.SI.F_max*ones(3,1); env.SI.tau_max*ones(3,1)];
        end
        
        function [Fc_opt, tau_joints] = solve(obj, sys, traj_des, t_idx, use_optimization)
            % Résolution QP ou contrôle direct
            %  sys.RL est cell array, sys.r_contact est [3x1] numeric
            
            % Ajout champs manquants
            if ~isfield(sys, 'time')
                sys.time = traj_des.t(t_idx);
            end
            
            % Gains structure
            gains = struct('Kr', obj.Kr, 'Dr', obj.Dr, 'Kb', obj.Kb, 'Db', obj.Db);
            
            %% Calcul wrenches désirés (éq. 12-16)
            [F_d_r, F_d_b] = compute_feedforward_wrench(sys, traj_des, t_idx, gains);
            
            %% Résolution
            if use_optimization
                Fc_opt = obj.solve_qp(F_d_r, F_d_b, sys);
            else
                % Non-optimisé: tracking robot uniquement
                g_cb = obj.build_transform(sys);
                Ad_gcb = adjoint_SE3(g_cb);
                Fc_opt = pinv(Ad_gcb') * F_d_r;
                Fc_opt = max(min(Fc_opt, obj.F_max), obj.F_min);
            end
            
            %% Mapping Fc -> tau_joints
            tau_joints = sys.Jc_joints' * Fc_opt;
        end
        
      function Fc_opt = solve_qp(obj, F_d_r, F_d_b, sys)
    % QP conforme équations (18-22)
    % ✅ AJOUT: Validation numérique et fallback robuste
    
    %% Validation entrées
    if any(isnan(F_d_r)) || any(isinf(F_d_r))
        warning('[QP] F_d_r contient NaN/Inf à t=%.2f', sys.time);
        Fc_opt = zeros(6,1);
        return;
    end
    if any(isnan(F_d_b)) || any(isinf(F_d_b))
        warning('[QP] F_d_b contient NaN/Inf à t=%.2f', sys.time);
        Fc_opt = zeros(6,1);
        return;
    end
    
    %% Transformation contact->base
    try
        g_cb = obj.build_transform(sys);
    catch ME
        warning('[QP] Erreur build_transform: %s', ME.message);
        Fc_opt = zeros(6,1);
        return;
    end
    
    Ad_gcb = adjoint_SE3(g_cb);
    
    % Validation Adjoint
    if any(isnan(Ad_gcb(:))) || any(isinf(Ad_gcb(:)))
        warning('[QP] Adjoint invalide à t=%.2f', sys.time);
        Fc_opt = zeros(6,1);
        return;
    end
    
    %% Résidus (éq. 19-21)
    A_r = Ad_gcb';
    b_r = F_d_r;
    
    A_b = Ad_gcb';
    b_b = -F_d_b;
    
    A_c = eye(6);
    b_c = zeros(6,1);
    
    %% Hessienne et gradient
    H = A_r'*obj.Qr*A_r + A_b'*obj.Qb*A_b + A_c'*obj.Qc*A_c;
    H = (H + H')/2;  % Symétrie
    
    % ✅ AJOUT: Régularisation si mal conditionnée
    cond_H = cond(H);
    if cond_H > 1e12 || any(isnan(H(:))) || any(isinf(H(:)))
        warning('[QP] H mal conditionnée (cond=%.2e) à t=%.2f', cond_H, sys.time);
        % Régularisation Tikhonov
        H = H + 1e-6 * eye(6);
    end
    
    f = -2*(b_r'*obj.Qr*A_r + b_b'*obj.Qb*A_b + b_c'*obj.Qc*A_c)';
    
    % Validation gradient
    if any(isnan(f)) || any(isinf(f))
        warning('[QP] Gradient invalide à t=%.2f', sys.time);
        Fc_opt = zeros(6,1);
        return;
    end
    
    %% Contraintes
    lb = obj.F_min;
    ub = obj.F_max;
    
    %% Résolution
    options = optimoptions('quadprog', 'Display', 'off', ...
                          'Algorithm', 'interior-point-convex', ...
                          'MaxIterations', 200, ...
                          'ConstraintTolerance', 1e-6, ...
                          'OptimalityTolerance', 1e-6);
    
    try
        [Fc_opt, ~, exitflag] = quadprog(H, f, [], [], [], [], lb, ub, [], options);
    catch ME
        warning('[QP] Exception quadprog: %s', ME.message);
        exitflag = -99;
    end
    
    %% Gestion échecs
    if exitflag < 0
        if exitflag ~= -99  % Ne pas re-logger si déjà loggé
            warning('[QP] Infeasible (exitflag=%d) at t=%.2f', exitflag, sys.time);
        end
        
        % Fallback: projection simple sur tracking robot
        Fc_opt = pinv(Ad_gcb', 1e-6) * F_d_r;  % Pseudoinverse avec tolérance
        
        % Saturation stricte
        Fc_opt = max(min(Fc_opt, ub), lb);
        
        % Vérification finale
        if any(isnan(Fc_opt)) || any(isinf(Fc_opt))
            warning('[QP] Fallback échoué, retour zéro');
            Fc_opt = zeros(6,1);
        end
    end
end
        function g_cb = build_transform(obj, sys)
            % Construit la transformation homogène contact->base
            % ✅ GÈRE LE FAIT QUE sys.RL EST CELL ARRAY
            %
            % g_cb = [R_cb, p_cb; 0 0 0 1]
            % où R_cb = RL{end} (rotation dernier link)
            %     p_cb = sys.r_contact (position contact)
            
            % Extraire rotation dernier link
            if iscell(sys.RL)
                % Version CasADi: cell array
                R_end = sys.RL{end};  % ✅ Accès cell array
            else
                % Version classique: 3D array (fallback)
                R_end = sys.RL(:,:,end);
            end
            
            % Position contact
            r_contact = sys.r_contact;
            
            % Vérifier dimensions
            if size(R_end, 1) ~= 3 || size(R_end, 2) ~= 3
                error('R_end doit être [3x3], reçu [%dx%d]', size(R_end,1), size(R_end,2));
            end
            if length(r_contact) ~= 3
                error('r_contact doit être [3x1], reçu [%dx%d]', size(r_contact,1), size(r_contact,2));
            end
            
            % Construction transformation homogène
            g_cb = [R_end,        r_contact(:);  % Force colonne
                    0, 0, 0,      1];
        end
    end
end