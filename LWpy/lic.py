import os
import struct
import hashlib

class read_stream:
    def __init__(self, fname, spline_dir='./'):
        f = open(fname, 'rb')
        self.data = f.read()
        f.close()
        self.spline_dir = spline_dir
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
        val = int.from_bytes(self.read_bin(n), endian, signed=signed)
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

    def read_block_header(self):
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
        final_type_0 = self.read_int(self.particle_size, signed=True)
        final_type_1 = self.read_int(self.particle_size, signed=True)
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
            name = os.path.join(self.spline_dir, name)
            if os.path.isfile(name):
                f = open(name, 'rb')
                check_data = f.read()
                f.close()
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
                "final_type_0": final_type_0,
                "final_type_1":  final_type_1,
                "totalCrossSection": txs_name,
                "differentialCrossSection": xs_name,
                "radius": radius,
                "height": height,
                }
        return d

    def read_ranged_block(self, version=1):
        d = self.read_volume_block(version=version)
        v = d["height"]
        del d["height"]
        d["length"] = v
        return d

    def read(self):
        blocks = []
        while self.pos < len(self.data):
            block_size, block_name, block_version = self.read_block_header()
            if block_name == 'EnumDef':
                block = self.read_enum_block(version=block_version)
            elif block_name == 'VolumeInjectionConfiguration':
                block = self.read_volume_block(version=block_version)
            elif block_name == 'RangedInjectionConfiguration':
                block = self.read_ranged_block(version=block_version)
            else:
                raise ValueError("Unrecognized block! " + block_name)
            blocks.append((block_name, block_version, block))
        return blocks


class write_stream:
    def __init__(self, fname, spline_dir='./'):
        self.fname = fname
        self.data = bytearray(0)
        self.spline_dir = spline_dir
        self.pos = 0
        self.size_size = 8
        self.enum_size_size = 4
        self.float_size = 4
        self.double_size = 8
        self.particle_size = 4

    def write_bin(self, data):
        self.data = self.data[:self.pos] + data + self.data[self.pos:]
        n = len(data)
        self.pos += n
        return n

    def write_int(self, val, n, endian='little', signed=False):
        data = val.to_bytes(n, endian, signed=signed)
        return self.write_bin(data)

    def write_float(self, val, endian='little', signed=True):
        if endian == 'little':
            e = '<'
        elif endian == 'big':
            e = '>'
        else:
            e = ''
        fmt = e + 'f'
        n = struct.calcsize(fmt)
        buf = bytearray(n)
        struct.pack_into(fmt, buf, 0, val)
        return self.write_bin(buf)

    def write_double(self, val, endian='little', signed=True):
        if endian == 'little':
            e = '<'
        elif endian == 'big':
            e = '>'
        else:
            e = ''
        fmt = e + 'd'
        n = struct.calcsize(fmt)
        buf = bytearray(n)
        struct.pack_into(fmt, buf, 0, val)
        return self.write_bin(buf)

    def write_string(self, val, endian='little', signed=False):
        n = self.write_int(len(val), self.size_size, signed=signed)
        n += self.write_bin(val.encode('ascii'))
        return n

    def write_block_header(self, block_size, block_name, block_version):
        n = self.write_int(block_size, self.size_size)
        n += self.write_string(block_name)
        n += self.write_int(block_version, 1)
        return n

    def write_block(self, block, writer):
        n = 0
        block_name, block_version, block, = block
        pos0 = self.pos
        block_size = writer(block, version=block_version)
        n += block_size
        self.pos = pos0
        n += self.write_block_header(block_size, block_name, block_version)
        self.pos += block_size
        return n

    def write_enum_block(self, block, version=1):
        n = 0
        name, enum = block
        n += self.write_string(name)
        enum_size = len(enum)
        n += self.write_int(enum_size, 4)
        enum_names = sorted(enum.keys())
        enum_vals = [enum[s] for s in enum_names]
        for enum_name, enum_val in zip(enum_names, enum_vals):
            n += self.write_int(enum_val, 8, signed=True)
            n += self.write_string(enum_name)
        return n

    def write_volume_block(self, block, version=1):
        n = 0
        events = block["events"]
        energy_min = block["energy_min"]
        energy_max = block["energy_max"]
        powerlaw_index = block["powerlaw_index"]
        azimuth_min = block["azimuth_min"]
        azimuth_max = block["azimuth_max"]
        zenith_min = block["zenith_min"]
        zenith_max = block["zenith_max"]
        final_type_0 = block["final_type_0"]
        final_type_1 = block["final_type_1"]
        totalCrossSection = block["totalCrossSection"]
        differentialCrossSection = block["differentialCrossSection"]
        radius = block["radius"]
        height = block["height"]

        txs_data = open(os.path.join(self.spline_dir, totalCrossSection), 'rb').read()
        xs_data = open(os.path.join(self.spline_dir, differentialCrossSection), 'rb').read()
        txs_size = len(txs_data)
        xs_size = len(xs_data)

        xs_hash = hashlib.sha512(xs_data).hexdigest()
        xs_name = xs_hash + '.fits'
        txs_hash = hashlib.sha512(txs_data).hexdigest()
        txs_name = txs_hash + '.fits'

        n += self.write_int(events, 4)
        n += self.write_double(energy_min)
        n += self.write_double(energy_max)
        n += self.write_double(powerlaw_index)
        n += self.write_double(azimuth_min)
        n += self.write_double(azimuth_max)
        n += self.write_double(zenith_min)
        n += self.write_double(zenith_max)
        n += self.write_int(final_type_0, self.particle_size, signed=True)
        n += self.write_int(final_type_1, self.particle_size, signed=True)
        n += self.write_int(xs_size, self.size_size)
        n += self.write_bin(xs_data)
        n += self.write_int(txs_size, self.size_size)
        n += self.write_bin(txs_data)
        n += self.write_double(radius)
        n += self.write_double(height)
        return n

    def write_ranged_block(self, block, version=1):
        v = block["length"]
        del block["length"]
        block["height"] = v
        return self.write_volume_block(block, version=version)

    def write(self, blocks):
        n = 0
        for block in blocks:
            block_name, block_vesion, block_data = block
            if block_name == 'EnumDef':
                n += self.write_block(block, self.write_enum_block)
            elif block_name == 'VolumeInjectionConfiguration':
                n += self.write_block(block, self.write_volume_block)
            elif block_name == 'RangedInjectionConfiguration':
                n += self.write_block(block, self.write_ranged_block)
            else:
                raise ValueError("Unrecognized block! " + block_name)
            pass
        f = open(self.fname, 'wb')
        f.write(bytes(self.data))
        f.close()
        return n
