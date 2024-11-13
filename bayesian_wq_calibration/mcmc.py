"""
Functions for Markov Chain Monte Carlo sampling for Bayesian calibration method.
"""



def decision_variables_to_dict(grouping, wall_coeffs):

    if grouping == 'single':
        wall_coeffs = {'single': wall_coeffs[0]}
    elif grouping == 'material':
        wall_coeffs = {
            'metallic': wall_coeffs[0],
            'cement': wall_coeffs[1],
            'plastic_unknown': wall_coeffs[2],
        }
    elif grouping == 'material-diameter':
        wall_coeffs = {
            'metallic_less_than_150': wall_coeffs[0],
            'metallic_greater_than_150': wall_coeffs[1],
            'cement': wall_coeffs[2],
            'plastic_unknown': wall_coeffs[3],
        }
    elif grouping == 'roughness':
        wall_coeffs = {
            'less_than_50': wall_coeffs[0],
            'between_50_and_65': wall_coeffs[1],
            'between_65_and_80': wall_coeffs[2],
            'between_80_and_100': wall_coeffs[3],
            'between_100_and_120': wall_coeffs[4],
            'greater_than_120': wall_coeffs[5],
        }
    else:
        raise ValueError('Wall grouping type is not valid. Please choose from: single, material, material-diameter, or roughness.')
    
    return wall_coeffs