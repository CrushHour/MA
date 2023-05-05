# %%
import numpy as np
import trimesh
from stl import mesh
import xml.etree.ElementTree as ET
import os
import json
import yaml
#import calculate_angels as ca


# settings
STL_PATH = './STL'
STL_SUB_PATH = 'convex_meshes'
SCALE = 0.001  # from mm to m
DENSITY_BONE = 1850  # kg / m**3
POS = [0, 0, 1.6]
POS = [p * SCALE * 1000 for p in POS]
REGENERATE_SUB_MESHES = False
MODIFY_TEMPLATE = False
CHANGE_SITE_POS = True

DAMPING = 200
STIFFNESS = 2000


names = ['femur', 'tibia', 'fibula', 'patella']

FILENAME = '../knee_model.xml'
TEMPLATE_NAME = '../knee_model_template.xml'

# root names
WORLDBODY = 'worldbody'
ASSET = 'asset'
TENDON = 'tendon'


def write_arr_xml(arr):
    """Converts an array of elements to a space-separated string."""
    return ' '.join(str(elem) for elem in arr)


def update_scale(root):
    """Updates the scale of the assets."""
    for elem in root.find(ASSET):
        elem.attrib['scale'] = write_arr_xml([SCALE, SCALE, SCALE])


def update_all_pos(root):
    """Updates the positions and quaternions of elements in the worldbody."""
    for elem in root.find(WORLDBODY):
        elem.attrib['pos'] = write_arr_xml(POS)
        elem.attrib['quat'] = write_arr_xml([1, 0, 0, 0])


def update_all_pos_quat(root, name, pos, quat):
    """Updates the position and quaternion of an element with the specified name."""
    for elem in root[3]:
        if elem.attrib['name'] == name:
            elem.attrib['pos'] = write_arr_xml(pos)
            elem.attrib['quat'] = write_arr_xml(quat)


def get_cog_femur(name='femur'):
    """Returns the center of gravity of a mesh."""
    loc_mesh = mesh.Mesh.from_file(f'{STL_PATH}/{name}.stl')
    loc_mesh.vectors = loc_mesh.vectors * SCALE
    _, _, cog, _ = loc_mesh.get_mass_properties_with_density(DENSITY_BONE)
    return cog


def update_inertias(root):
    """Updates the mass, center of gravity, and inertia tensor of each element."""
    for name in names:
        loc_mesh = mesh.Mesh.from_file(f'{STL_PATH}/{name}.stl')
        loc_mesh.vectors = loc_mesh.vectors * SCALE
        _, vmass, cog, inertia = loc_mesh.get_mass_properties_with_density(
            DENSITY_BONE)
        inertia_arr = [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                       inertia[0, 1], inertia[0, 2], inertia[1, 2]]

        for elem in root.find(WORLDBODY):
            recursive_rooting(elem, name, cog, vmass, inertia_arr)


def recursive_rooting(elem, name, cog, vmass, inertia_arr):
    if 'name' in elem.attrib.keys() and elem.attrib['name'] == name:
        print(name)
        for loc_elem in elem:
            if 'mass' in loc_elem.attrib.keys():
                loc_elem.attrib['mass'] = f'{vmass}'
                loc_elem.attrib['pos'] = write_arr_xml(cog)
                loc_elem.attrib['fullinertia'] = write_arr_xml(inertia_arr)
        return

    for loc_elem in elem:
        recursive_rooting(loc_elem, name, cog, vmass, inertia_arr)


def recursive_site(elem, name, tendon_name, pos):
    if 'name' in elem.attrib.keys() and elem.attrib['name'] == name:
        for loc_elem in elem:
            if 'name' in loc_elem.attrib.keys() and loc_elem.attrib['name'] == tendon_name:
                loc_elem.attrib['pos'] = write_arr_xml(pos)
                loc_elem.attrib['type'] = 'sphere'
                loc_elem.attrib['size'] = f'{SCALE}'
                return
        # we looped over all so the site does not exist yet
        attrib = {
            'name': tendon_name,
            'pos': write_arr_xml(pos),
            'type': 'sphere',
            'size': f'{SCALE}',
        }
        elem.makeelement('site', attrib)
        ET.SubElement(elem, 'site', attrib)

    for loc_elem in elem:
        recursive_site(loc_elem, name, tendon_name, pos)


def recursive_mesh(elem, name, mesh_name):
    if 'name' in elem.attrib.keys() and elem.attrib['name'] == name:
        print(name)
        for loc_elem in elem:
            if 'name' in loc_elem.attrib.keys() and loc_elem.attrib['name'] == mesh_name:
                loc_elem.attrib['type'] = 'mesh'
                loc_elem.attrib['class'] = 'collision'
                return
        # we looped over all so the site does not exist yet
        attrib = {
            'name': mesh_name,
            'mesh': mesh_name,
            'type': 'mesh',
            'class': 'collision',
        }
        elem.makeelement('geom', attrib)
        ET.SubElement(elem, 'geom', attrib)

    for loc_elem in elem:
        recursive_mesh(loc_elem, name, mesh_name)


def add_zylinders(root, zyl_path='../assets/wrap'):
    names = ['femur', 'tibia']
    wrap_files = os.listdir(zyl_path)

    for wrap_file in wrap_files:
        with open(f'{zyl_path}/{wrap_file}') as jsonfile:
            data = json.load(jsonfile)

        # extract point infos
        point_data = data['markups'][0]['controlPoints']
        points = [point['position'] for point in point_data]

        z1 = points[0][2]
        z2 = points[1][2]
        points[0] = [p * SCALE for p in points[0]]
        points[1] = [p * SCALE for p in points[1]]

        if z1 > z2:
            point1 = points[0]
            point2 = points[1]
        else:
            point1 = points[1]
            point2 = points[0]

        tendon_name = wrap_file.split('.')[0]

        # now search the tree and add site:
        st_name = f'{tendon_name}'
        recursive_zyl(root, names[1], st_name, point1, point2)


def recursive_zyl(elem, name, mesh_name, p1, p2):
    string_name1 = write_arr_xml(p1)
    string_name2 = write_arr_xml(p2)
    fromto = f'{string_name1} {string_name2}'

    if 'name' in elem.attrib.keys() and elem.attrib['name'] == name:
        print(name)
        for loc_elem in elem:
            if 'name' in loc_elem.attrib.keys() and loc_elem.attrib['name'] == mesh_name:
                loc_elem.attrib['type'] = 'cylinder'
                loc_elem.attrib['size'] = '0.025'
                loc_elem.attrib['rgba'] = '.2 .2 .5 0.0'
                loc_elem.attrib['fromto'] = fromto
                loc_elem.attrib['class'] = 'visual'
                return
        # we looped over all so the site does not exist yet
        attrib = {
            'name': mesh_name,
            'type': 'cylinder',
            'size': '0.025',
            'rgba': '.2 .2 .5 0.0',
            'fromto': fromto,
            'class': 'visual'
        }
        elem.makeelement('geom', attrib)
        ET.SubElement(elem, 'geom', attrib)

    for loc_elem in elem:
        recursive_zyl(loc_elem, name, mesh_name, p1, p2)


def load_tendons(root, tendon_path='../assets/tendons'):
    """Loads tendon data from JSON files and adds them to the model."""
    names = ['femur', 'tibia']

    tendon_files = os.listdir(tendon_path)
    root.remove(root.find(TENDON))
    root.makeelement(TENDON, {})
    ET.SubElement(root, TENDON, {})

    yaml_dict = {}

    zyl_path = '../assets/wrap'
    wrap_files = os.listdir(zyl_path)
    wrapped_files = [wrap_file.split('.')[0].split('_')[0]
                     for wrap_file in wrap_files]

    for tendon in tendon_files:
        with open(f'{tendon_path}/{tendon}') as jsonfile:
            data = json.load(jsonfile)

        point_data = data['markups'][0]['controlPoints']
        points = [point['position'] for point in point_data]

        z1, z2 = points[0][2], points[1][2]
        points[0] = [p * SCALE for p in points[0]]
        points[1] = [p * SCALE for p in points[1]]

        point1, point2 = (points[0], points[1]) if z1 > z2 else (
            points[1], points[0])

        tendon_name = tendon.split('.')[0]

        st_name, end_name = f'{tendon_name}_start', f'{tendon_name}_end'

        dist = np.linalg.norm(np.array(point1) - np.array(point2))

        loc_attrib = {
            'name': tendon_name,
            'stiffness': f"{STIFFNESS * SCALE * 1000}",
            'damping': f"{DAMPING}",
            'springlength': f'{dist}',
            'width': f'{SCALE * 2}',
            'rgba': "0.9 0.2 0.2 0.9",
        }

        if MODIFY_TEMPLATE:
            loc_attrib = {
                'name': tendon_name,
                'stiffness': "{{" + f'{tendon_name}.stiffness' + '}}',
                'damping': "{{" + f'{tendon_name}.damping' + '}}',
                'springlength': "{{" + f'{tendon_name}.springlength' + '}}',
                'width': f'{SCALE * 2}',
                'rgba': "0.9 0.2 0.2 0.9",
            }
        
        yaml_dict[tendon_name] = {
            'stiffness': f"{STIFFNESS * SCALE * 1000}",
            'damping': f"{DAMPING}",
            'springlength': f'{dist}',
        }


        # add the start and end position in the xml
        if CHANGE_SITE_POS:
            point1_name = [
                '{{' + f'{tendon_name}.pos_start.x' + '}}' + ' '
                '{{' + f'{tendon_name}.pos_start.y' + '}}' + ' '
                '{{' + f'{tendon_name}.pos_start.z' + '}}'
            ]

            point2_name = [
                '{{' + f'{tendon_name}.pos_end.x' + '}}' + ' '
                '{{' + f'{tendon_name}.pos_end.y' + '}}' + ' '
                '{{' + f'{tendon_name}.pos_end.z' + '}}'
            ]

            recursive_site(root, names[0], st_name, point1_name)
            recursive_site(root, names[1], end_name, point2_name)


            yaml_dict[tendon_name]['pos_start'] = {
                'x' : f'{point1[0]}',
                'y' : f'{point1[1]}',
                'z' : f'{point1[2]}'
            }

            yaml_dict[tendon_name]['pos_end'] = {
                'x' : f'{point2[0]}',
                'y' : f'{point2[1]}',
                'z' : f'{point2[2]}',
            }

        else:
            recursive_site(root, names[0], st_name, point1)
            recursive_site(root, names[1], end_name, point2)

        

        tendon_root = root.find(TENDON)
        tendon_root.makeelement('spatial', loc_attrib)
        ET.SubElement(tendon_root, 'spatial', loc_attrib)

        for pos_tendon in tendon_root:
            if pos_tendon.attrib['name'] == tendon_name:
                pos_tendon.makeelement('site', {'site': st_name})
                pos_tendon.makeelement('site', {'site': end_name})
                ET.SubElement(pos_tendon, 'site', {'site': st_name})
                if tendon_name in wrapped_files:
                    ET.SubElement(pos_tendon, 'geom', {
                                  'geom': f'{tendon_name}_wrap'})
                ET.SubElement(pos_tendon, 'site', {'site': end_name})

    # Save the yaml as well:
    yaml_dict['tibia'] = {
        'dx': '0.0',
        'dy': '0.0',
        'dz': '0.0',
        'dqw': '0.0',
        'dqx': '0.0',
        'dqy': '0.0',
        'dqz': '0.0',
    }

    with open('../params/parameters_init.yaml', 'w') as f:
        yaml.dump(yaml_dict, f)


def update_from_dict(dict):
    tree = ET.parse(FILENAME)
    root = tree.getroot()
    update_all_pos_quat(root, 'femur', dict['femur_pos'], dict['femur_quat'])
    update_all_pos_quat(root, 'tibia', dict['tibia_pos'], dict['tibia_quat'])
    tree.write(FILENAME)

    tree = ET.parse(TEMPLATE_NAME)
    root = tree.getroot()
    update_all_pos_quat(root, 'femur', dict['femur_pos'], dict['femur_quat'])
    update_all_pos_quat(root, 'tibia', dict['tibia_pos'], dict['tibia_quat'])
    tree.write(TEMPLATE_NAME)


def prepare_all():
    tree = ET.parse(FILENAME)
    root = tree.getroot()
    load_tendons(root)
    add_zylinders(root)
    update_all_pos(root)
    update_inertias(root)

    if REGENERATE_SUB_MESHES:
        generate_convex_sub_meshes(root)

    update_scale(root)
    tree.write(FILENAME)


# %%
if __name__ == '_main_':
    prepare_all()
    MODIFY_TEMPLATE = True
    if MODIFY_TEMPLATE:
        FILENAME = './tendom_finger_template.xml'
    prepare_all()

# %%