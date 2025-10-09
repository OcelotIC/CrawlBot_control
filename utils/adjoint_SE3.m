function Ad_g = adjoint_SE3(g)
% Calcul Adjoint SE(3) conforme équation (4.8)
%
% g = [R, p; 0 0 0 1] ∈ SE(3)
%
% Ad_g = [ R,      [p]×R ]
%        [ 0,      R     ]

R = g(1:3, 1:3);
p = g(1:3, 4);

% Skew-symmetric matrix [p]×
p_skew = [    0, -p(3),  p(2);
           p(3),     0, -p(1);
          -p(2),  p(1),     0];

% Construction Adjoint
Ad_g = [R,           p_skew * R;
        zeros(3,3),  R        ];

end