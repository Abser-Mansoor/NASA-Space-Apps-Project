import numpy as np
def compute_rh2m(T2M, QV2M, PS):
    """
    Compute 2m Relative Humidity (RH2M) from MERRA-2 fields.
    
    Parameters:
        T2M  : 2m temperature [K]
        QV2M : 2m specific humidity [kg/kg]
        PS   : surface pressure [Pa]
        
    Returns:
        RH2M : 2m relative humidity [%]
    """
    # Vapor pressure (Pa)
    e = (QV2M * PS) / (0.622 + 0.378 * QV2M)

    # Saturation vapor pressure (Pa) - Tetens formula
    T_c = T2M - 273.15  # convert to °C
    e_s = 610.94 * np.exp((17.625 * T_c) / (T_c + 243.04))

    # Relative Humidity (%)
    RH = (e / e_s) * 100.0

    # Clamp values to [0, 100] for physical realism
    RH = np.clip(RH, 0, 100)

    return RH