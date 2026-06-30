#!/usr/bin/env python3
"""Read-only diagnosis: WHERE the torso orientation-reference chain breaks continuity.
Uses the canonical run's sim_log (q_torso = realized R; q_torso_ref = controller reference R;
e_torso_ori = the logged metric). All quaternions wxyz. geodesic angle in deg.
"""
import json, sys
import numpy as np
import pinocchio as pin

RUN = sys.argv[1] if len(sys.argv) > 1 else 'results/figA_canon'
sl = json.load(open(f'{RUN}/sim_log.json'))
ph = np.array(sl['phase']); step = np.array(sl['step_idx'], int)
t = np.array(sl['t'], float)
qt = np.array(sl['q_torso'], float)        # realized torso R
qr = np.array(sl['q_torso_ref'], float)    # controller reference R (what reference_at returned)
eto = np.array(sl['e_torso_ori'], float)   # logged metric
n = len(t)

def R(q): w,x,y,z = q; return pin.Quaternion(w,x,y,z).toRotationMatrix()
def geo(A,B): return float(np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1))))
def rpy(q): return np.degrees(pin.rpy.matrixToRpy(R(q)))

print(f'RUN {RUN}  n={n}  phases={sorted(set(ph))}  steps={sorted(set(step.tolist()))}')

# contiguous (phase) blocks
blocks=[]
for i in range(n):
    if blocks and blocks[-1][0]==ph[i]: blocks[-1][2]=i
    else: blocks.append([ph[i], i, i])
print(f'\n{len(blocks)} phase blocks: ' + ' '.join(f'{b[0]}[{b[1]}-{b[2]}]' for b in blocks))

print('\n=== POINT 2: per DS->SS transition — reference jump vs torso continuity vs metric jump ===')
print(' transition (lastDS -> firstSS) | ref_jump deg | torso_jump deg | e_ori last_DS -> first_SS')
for i in range(1, n):
    if ph[i-1]=='DS' and ph[i]=='SS':
        rj = geo(R(qr[i-1]), R(qr[i]))     # does the REFERENCE jump?
        tj = geo(R(qt[i-1]), R(qt[i]))     # is the TORSO continuous?
        print(f'  tick {i-1:4d}->{i:<4d} (t={t[i]:6.2f}s, step {step[i]}) | '
              f'{rj:8.4f}     | {tj:8.4f}      | {eto[i-1]:6.3f} -> {eto[i]:.3f}')

print('\n=== in-DS drift: realized torso R rotates during each DS block (ref was fixed at block entry) ===')
print(' DS block [ticks]   span s | torso drift entry->exit deg | e_ori entry->exit')
for b in blocks:
    if b[0]!='DS': continue
    lo,hi=b[1],b[2]
    drift = geo(R(qt[lo]), R(qt[hi]))
    print(f'  [{lo:4d}-{hi:4d}] {t[hi]-t[lo]:5.2f}s | {drift:8.4f}                    | {eto[lo]:.3f} -> {eto[hi]:.3f}')

print('\n=== SS SLERP span: does the SS orientation reference ADVANCE (R_t0->R_t1) or stay ~constant? ===')
print(' SS block [ticks] step | SLERP span R_ref(start)->R_ref(end) deg | torso advance deg')
for b in blocks:
    if b[0]!='SS': continue
    lo,hi=b[1],b[2]
    span = geo(R(qr[lo]), R(qr[hi]))
    tadv = geo(R(qt[lo]), R(qt[hi]))
    print(f'  [{lo:4d}-{hi:4d}] s{step[lo]} | {span:8.4f}                              | {tadv:8.4f}')

print('\n=== POINT 3: R_dock(k) = controller reference R at SS-block END (= R_t1, the dock-IK target) ===')
print(' step | R_dock rpy (deg)            | step-to-step angle R_dock(k-1)->R_dock(k)')
prev=None
for b in blocks:
    if b[0]!='SS': continue
    hi=b[2]; k=step[hi]
    r=rpy(qr[hi])
    stepang = '' if prev is None else f'{geo(R(prev), R(qr[hi])):.4f} deg'
    print(f'   {k:2d}  | roll {r[0]:6.2f} pitch {r[1]:6.2f} yaw {r[2]:6.2f} | {stepang}')
    prev=qr[hi]

print('\n=== metric sawtooth summary (e_torso_ori, the logged C-audit metric) ===')
ss=ph=='SS'; ds=ph=='DS'
print(f'  SS e_ori: first-tick-of-each-SS-block values = ', end='')
print([round(eto[b[1]],3) for b in blocks if b[0]=='SS'])
print(f'  DS e_ori: last-tick-of-each-DS-block values  = ', end='')
print([round(eto[b[2]],3) for b in blocks if b[0]=='DS'])
