import numpy as np

NDIMS = 3
NCOLORS = 4
NUM_PLAQS = NDIMS * (NDIMS - 1) / 2
NUM_TPLAQS = NDIMS - 1
NUM_SPLAQS = NUM_PLAQS - NUM_TPLAQS

beta = 6
cfg_num = 200
lattice_size = [12, 12, 12]
volume = np.prod(lattice_size)
conf_name = f"./data/su{NCOLORS}_nd{NDIMS}_beta{beta}_L{lattice_size[0]}_T{lattice_size[NDIMS-1]}_cfg_{cfg_num}.bin"
conf = np.fromfile(conf_name, dtype=np.complex128).reshape(
    (volume, NDIMS, NCOLORS, NCOLORS)
)

def check_unitary(conf, prec=1e-15):
    success = True
    for i in range(volume):
        for mu in range(NDIMS):
            unitary = conf[i, mu, ...] @ conf[i, mu, ...].conj().transpose()
            check = unitary - np.identity(NCOLORS)
            if np.max(np.abs(check)) > prec:
                success = False
    return success


def index_nd_nm(id: int):
    x = []
    temp = id
    for i in range(NDIMS):
        x.append(temp % lattice_size[i])
        temp = temp // lattice_size[i]

    return x


def index_nd_nm_coord(x: list):
    index = 0
    factor = 1
    for i in range(NDIMS):
        index += x[i] * factor
        factor *= lattice_size[i]

    return index


def calc_plaq(conf):
    plaq_s = 0
    plaq_t = 0
    for i in range(volume):
        x_org = index_nd_nm(i)
        for mu in range(NDIMS):
            xmu = x_org.copy()
            xmu[mu] = (xmu[mu] + 1) % lattice_size[mu]
            imu = index_nd_nm_coord(xmu)
            for nu in range(mu + 1, NDIMS):
                xnu = x_org.copy()
                xnu[nu] = (xnu[nu] + 1) % lattice_size[nu]
                inu = index_nd_nm_coord(xnu)
                # print(f"i={i},mu={mu},nu={nu},x={xmu},x_org={xnu}")
                plaq_multi = (
                    conf[i, mu, ...]
                    @ conf[imu, nu, ...]
                    @ conf[inu, mu, ...].conj().transpose()
                    @ conf[i, nu, ...].conj().transpose()
                )
                if nu == NDIMS - 1:
                    plaq_t += np.real(np.trace(plaq_multi))
                else:
                    plaq_s += np.real(np.trace(plaq_multi))
    plaq_s /= NUM_SPLAQS * NCOLORS * volume
    plaq_t /= NUM_TPLAQS * NCOLORS * volume

    return (plaq_s, plaq_t)


if __name__ == "__main__":
    if check_unitary(conf):
        print(f"The configuration '{conf_name}' satisfies unitary condition")
    else:
        raise ValueError(
            f"The configuration '{conf_name}' is not unitary, please check the data!"
        )
    
    plaq_s, plaq_t = calc_plaq(conf)
    print(f"Spatial plaquettes = {plaq_s}, temporal plaquettes = {plaq_t}")
