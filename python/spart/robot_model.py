"""Robot model: URDF parser and connectivity map.

Translated from urdf2robot.m, DH_Serial2robot.m, ConnectivityMap.m.
"""

import numpy as np
import xml.etree.ElementTree as ET
from .attitude import angles321_dcm


def connectivity_map(robot):
    """Compute branch connectivity and child maps.

    Parameters
    ----------
    robot : dict – Robot model (must have 'joints', 'links', 'n_links_joints').

    Returns
    -------
    branch : (n, n) array – branch[i, j] == 1 if link i and j are on same branch.
    child : (n, n) array – child[i, j] == 1 if link i is a child of link j.
    child_base : (n,) array – child_base[i] == 1 if link i connects to base.
    """
    n = robot['n_links_joints']

    # Branch connectivity
    branch = np.zeros((n, n), dtype=int)
    for i in range(n - 1, -1, -1):
        last_parent = i
        branch[i, i] = 1
        while True:
            parent_joint_idx = robot['links'][last_parent]['parent_joint'] - 1
            parent_link = robot['joints'][parent_joint_idx]['parent_link']
            if parent_link == 0:
                break
            parent_link_idx = parent_link - 1
            branch[i, parent_link_idx] = 1
            last_parent = parent_link_idx

    # Child map
    child = np.zeros((n, n), dtype=int)
    child_base = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        parent_joint_idx = robot['links'][i]['parent_joint'] - 1
        parent_link = robot['joints'][parent_joint_idx]['parent_link']
        if parent_link != 0:
            child[i, parent_link - 1] = 1
        else:
            child_base[i] = 1

    return branch, child, child_base


def urdf2robot(filename, verbose=False):
    """Create a SPART robot model from a URDF file.

    Parameters
    ----------
    filename : str – Path to the URDF file.
    verbose : bool – Print verbose output.

    Returns
    -------
    robot : dict – Robot model dictionary.
    robot_keys : dict – Name-to-ID maps for links, joints, q.
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    if root.tag != 'robot':
        robot_elem = root.find('robot')
        if robot_elem is None:
            raise ValueError("URDF must contain a <robot> element")
    else:
        robot_elem = root

    robot_name = robot_elem.get('name', 'unnamed')

    # Parse links
    link_elements = robot_elem.findall('link')
    # Parse joints (only direct children)
    joint_elements = robot_elem.findall('joint')

    n_total_links = len(link_elements)

    if verbose:
        print(f"Number of links: {n_total_links} (including base)")

    # Build link map
    links_map = {}
    for link_xml in link_elements:
        link = {
            'name': link_xml.get('name'),
            'T': np.eye(4),
            'parent_joint': [],
            'child_joint': [],
            'mass': 0.0,
            'inertia': np.zeros((3, 3)),
        }

        inertial = link_xml.find('inertial')
        if inertial is not None:
            origin = inertial.find('origin')
            if origin is not None:
                xyz = origin.get('xyz')
                if xyz:
                    link['T'][:3, 3] = np.array([float(x) for x in xyz.split()])
                rpy = origin.get('rpy')
                if rpy:
                    rpy_vals = np.array([float(x) for x in rpy.split()])
                    link['T'][:3, :3] = angles321_dcm(rpy_vals).T

            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                link['mass'] = float(mass_elem.get('value', '0'))

            inertia_elem = inertial.find('inertia')
            if inertia_elem is not None:
                ixx = float(inertia_elem.get('ixx', '0'))
                iyy = float(inertia_elem.get('iyy', '0'))
                izz = float(inertia_elem.get('izz', '0'))
                ixy = float(inertia_elem.get('ixy', '0'))
                iyz = float(inertia_elem.get('iyz', '0'))
                ixz = float(inertia_elem.get('ixz', '0'))
                link['inertia'] = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz],
                ])

        links_map[link['name']] = link

    # Build joint map
    joints_map = {}
    for joint_xml in joint_elements:
        joint = {
            'name': joint_xml.get('name'),
            'type_name': joint_xml.get('type'),
            'parent_link': '',
            'child_link': '',
            'T': np.eye(4),
            'axis': np.array([0.0, 0.0, 0.0]),
        }

        type_name = joint['type_name']
        if type_name in ('revolute', 'continuous'):
            joint['type'] = 1
        elif type_name == 'prismatic':
            joint['type'] = 2
        elif type_name == 'fixed':
            joint['type'] = 0
            joint['axis'] = np.array([0.0, 0.0, 0.0])
        else:
            raise ValueError(f"Joint type '{type_name}' not supported.")

        # Origin
        origin = joint_xml.find('origin')
        if origin is not None:
            xyz = origin.get('xyz')
            if xyz:
                joint['T'][:3, 3] = np.array([float(x) for x in xyz.split()])
            rpy = origin.get('rpy')
            if rpy:
                rpy_vals = np.array([float(x) for x in rpy.split()])
                joint['T'][:3, :3] = angles321_dcm(rpy_vals).T

        # Axis
        axis_elem = joint_xml.find('axis')
        if axis_elem is not None:
            joint['axis'] = np.array([float(x) for x in axis_elem.get('xyz').split()])
        elif joint['type'] != 0:
            raise ValueError(f"Joint '{joint['name']}' is moving and requires an axis.")

        # Parent link
        parent_elem = joint_xml.find('parent')
        if parent_elem is not None:
            joint['parent_link'] = parent_elem.get('link')
            parent = links_map[joint['parent_link']]
            parent['child_joint'].append(joint['name'])

        # Child link
        child_elem = joint_xml.find('child')
        if child_elem is not None:
            joint['child_link'] = child_elem.get('link')
            child_link = links_map[joint['child_link']]
            child_link['parent_joint'].append(joint['name'])

        # Correct transform: from parent link inertial frame
        parent = links_map[joint['parent_link']]
        joint['T'] = np.linalg.solve(parent['T'], joint['T'])

        joints_map[joint['name']] = joint

    # Find base link (link with no parent joint)
    base_link_name = None
    for name, link in links_map.items():
        if len(link['parent_joint']) == 0:
            base_link_name = name
            break

    if base_link_name is None:
        raise ValueError("Robot has no single base link!")

    if verbose:
        print(f"Base link: {base_link_name}")

    # Build robot structure
    robot = {
        'name': robot_name,
        'n_links_joints': n_total_links - 1,  # Exclude base
        'n_q': 0,
        'links': [],
        'joints': [],
        'base_link': {
            'mass': links_map[base_link_name]['mass'],
            'inertia': links_map[base_link_name]['inertia'],
        },
    }

    robot_keys = {
        'link_id': {base_link_name: 0},
        'joint_id': {},
        'q_id': {},
    }

    # Recursively add links and joints
    base_link = links_map[base_link_name]
    nl = -1
    nj = -1
    nq = 1

    for child_joint_name in base_link['child_joint']:
        child_joint = joints_map[child_joint_name]
        robot, robot_keys, nl, nj, nq = _urdf2robot_recursive(
            robot, robot_keys, links_map, joints_map,
            child_joint, nl + 1, nj + 1, nq,
        )

    robot['n_q'] = nq - 1

    if verbose:
        print(f"Number of joint variables: {robot['n_q']}")

    # Add connectivity map
    branch, child, child_base = connectivity_map(robot)
    robot['con'] = {
        'branch': branch,
        'child': child,
        'child_base': child_base,
    }

    return robot, robot_keys


def _urdf2robot_recursive(robot, robot_keys, links_map, joints_map,
                           child_joint, nl, nj, nq):
    """Recursively add joints and links to the robot structure."""
    # Joint
    joint = {
        'id': nj + 1,
        'type': child_joint['type'],
        'parent_link': robot_keys['link_id'][child_joint['parent_link']],
        'child_link': nl + 1,
        'axis': child_joint['axis'].copy(),
        'T': child_joint['T'].copy(),
    }

    if child_joint['type'] != 0:
        joint['q_id'] = nq
        robot_keys['q_id'][child_joint['name']] = nq
        nq += 1
    else:
        joint['q_id'] = -1

    # Extend lists if needed
    while len(robot['joints']) <= nj:
        robot['joints'].append(None)
    robot['joints'][nj] = joint

    # Link
    clink = links_map[child_joint['child_link']]
    link = {
        'id': nl + 1,
        'parent_joint': nj + 1,
        'T': clink['T'].copy(),
        'mass': clink['mass'],
        'inertia': clink['inertia'].copy(),
    }

    while len(robot['links']) <= nl:
        robot['links'].append(None)
    robot['links'][nl] = link

    # Store IDs
    robot_keys['joint_id'][child_joint['name']] = nj + 1
    robot_keys['link_id'][clink['name']] = nl + 1

    # Recurse into children
    for child_joint_name in clink['child_joint']:
        cj = joints_map[child_joint_name]
        robot, robot_keys, nl, nj, nq = _urdf2robot_recursive(
            robot, robot_keys, links_map, joints_map,
            cj, nl + 1, nj + 1, nq,
        )

    return robot, robot_keys, nl, nj, nq


def dh_serial2robot(DH_data):
    """Create a SPART robot model from DH parameters.

    Parameters
    ----------
    DH_data : dict – DH parameter data with keys 'n', 'base', 'man', 'EE'.

    Returns
    -------
    robot : dict – Robot model.
    T_Ln_EE : (4, 4) array – Transform from last link to end-effector.
    """
    from .kinematics import _dh_transform

    robot = {
        'name': DH_data.get('name', 'DH robot'),
        'n_links_joints': DH_data['n'],
        'n_q': DH_data['n'],
        'links': [],
        'joints': [],
        'base_link': {
            'mass': DH_data['base']['mass'],
            'inertia': DH_data['base']['I'],
        },
        'origin': 'DH',
    }

    for i in range(DH_data['n']):
        man = DH_data['man'][i]

        # Link
        link = {
            'id': i + 1,
            'parent_joint': i + 1,
            'mass': man['mass'],
            'inertia': man['I'],
        }
        # Compute T for link (simplified)
        link['T'] = np.eye(4)  # Placeholder
        robot['links'].append(link)

        # Joint
        joint = {
            'id': i + 1,
            'type': man['type'],
            'q_id': i + 1,
            'parent_link': i,
            'child_link': i + 1,
            'axis': np.array([0.0, 0.0, 1.0]),
        }
        if i == 0:
            joint['T'] = DH_data['base']['T_L0_J1']
        else:
            joint['T'] = np.eye(4)
            joint['T'][:3, 3] = DH_data['man'][i - 1]['b']
        robot['joints'].append(joint)

    # Connectivity
    branch, child, child_base = connectivity_map(robot)
    robot['con'] = {
        'branch': branch,
        'child': child,
        'child_base': child_base,
    }

    T_Ln_EE = np.eye(4)
    T_Ln_EE[:3, 3] = DH_data['man'][-1]['b']

    return robot, T_Ln_EE
