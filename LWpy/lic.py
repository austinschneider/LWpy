import os
import struct
import hashlib

class stream:
    def __init__(self, fname):
        self.data = open(fname, 'rb').read()
        self.pos = 0
        self.size_size = 8
        self.enum_size_size = 4
        self.float_size = 4
        self.double_size = 8
        self.particle_size = 4

    def read_bin(self, n):
        v = self.data[self.pos:self.pos+n]
        self.pos += n
        return v

    def read_int(self, n, endian='little', signed=False):
        val = int.from_bytes(self.read_bin(n), 'little', signed=signed)
        return val

    def read_float(self, endian='little', signed=True):
        if endian == 'little':
            e = '<'
        elif endian == 'big':
            e = '>'
        else:
            e = ''
        d = self.read_bin(self.float_size)
        v = struct.unpack(e+'f', d)[0]
        return v

    def read_double(self, endian='little', signed=True):
        if endian == 'little':
            e = '<'
        elif endian == 'big':
            e = '>'
        else:
            e = ''
        d = self.read_bin(self.double_size)
        v = struct.unpack(e+'d', d)[0]
        return v

    def read_string(self, endian='little', signed=False):
        v = self.read_int(self.size_size, signed=signed)
        s = self.read_bin(v).decode('ascii')
        return s

    def read_block(self):
        block_size = self.read_int(self.size_size)
        block_name = self.read_string()
        block_version = self.read_int(1)
        return block_size, block_name, block_version

    def read_enum_block(self, version=1):
        name = self.read_string()
        enum_size = self.read_int(4)
        enum_vals = []
        enum_names = []
        for i in range(enum_size):
            enum_v = self.read_int(8, signed=True)
            enum_vals.append(enum_v)
            enum_name = self.read_string()
            enum_names.append(enum_name)
        return name, dict(zip(enum_names,enum_vals))

    def read_volume_block(self, version=1):
        events = self.read_int(4)
        energy_min = self.read_double()
        energy_max = self.read_double()
        powerlaw_index = self.read_double()
        azimuth_min = self.read_double()
        azimuth_max = self.read_double()
        zenith_min = self.read_double()
        zenith_max = self.read_double()
        final_type_1 = self.read_int(self.particle_size, signed=True)
        final_type_2 = self.read_int(self.particle_size, signed=True)
        xs_size = self.read_int(self.size_size)
        xs_data = self.read_bin(xs_size)
        txs_size = self.read_int(self.size_size)
        txs_data = self.read_bin(txs_size)
        radius = self.read_double()
        height = self.read_double()
        xs_hash = hashlib.sha512(xs_data).hexdigest()
        xs_name = xs_hash + '.fits'
        txs_hash = hashlib.sha512(txs_data).hexdigest()
        txs_name = txs_hash + '.fits'

        for name, x_hash, data in [[xs_name, xs_hash, xs_data], [txs_name, txs_hash, txs_data]]:
            check = False
            if os.path.isfile(name):
                check_data = open(name, 'rb').read()
                check_hash = hashlib.sha512(data).hexdigest()
                check = x_hash == check_hash
            if not check:
                open(name, 'wb').write(data)

        d = {
            "events": events,
            "energy_min": energy_min,
            "energy_max": energy_max,
            "powerlaw_index": powerlaw_index,
            "azimuth_min": azimuth_min,
            "azimuth_max": azimuth_max,
            "zenith_min": zenith_min,
            "zenith_max": zenith_max,
            "final_type_1": final_type_1,
            "final_type_2":  final_type_2,
            "totalCrossSection": txs_name,
            "differentialCrossSection": xs_name,
            "radius": radius,
            "height": height,
        }
        return "volume", d

    def read_ranged_block(self, version=1):
        s, d = self.read_volume_block()
        v = d["height"]
        del d["height"]
        d["length"] = v
        return "ranged", d

    def read(self):
        blocks = []
        while self.pos < len(self.data):
            block_size, block_name, block_version = self.read_block()
            if block_name == 'EnumDef':
                block = self.read_enum_block(version=block_version)
            elif block_name == 'VolumeInjectionConfiguration':
                block = self.read_volume_block(version=block_version)
            elif block_name == 'RangedInjectionConfiguration':
                block = self.read_ranged_block(version=block_version)
            else:
                raise ValueError("Unrecognized block! " + block_name)
            blocks.append(block)
        return blocks

