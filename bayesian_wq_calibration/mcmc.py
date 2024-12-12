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
    elif grouping == 'material-velocity':
        wall_coeffs = {
            'metallic_low_velocity': wall_coeffs[0],
            'metallic_high_velocity': wall_coeffs[1],
            'cement_low_velocity': wall_coeffs[2],
            'cement_high_velocity': wall_coeffs[3],
            'plastic_low_velocity': wall_coeffs[4],
            'plastic_high_velocity': wall_coeffs[5],
        }
    else:
        raise ValueError('Wall grouping type is not valid. Please choose from: single, material, material-diameter, or material-velocity.')
    
    return wall_coeffs