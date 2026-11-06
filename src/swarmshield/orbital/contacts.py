import math
from datetime import UTC, datetime, timedelta

from .checkpoint import atomic_json
from .config import project_path


def generate(cfg):
    from sgp4.api import WGS72, Satrec
    from skyfield.api import EarthSatellite, load, wgs84

    epoch = datetime.fromisoformat(cfg.orbit.epoch)
    if epoch.tzinfo is None:
        raise ValueError("Orbital epoch needs a UTC offset")
    epoch = epoch.astimezone(UTC)
    ts = load.timescale(builtin=True)
    start = ts.from_datetime(epoch)
    end = ts.from_datetime(epoch + timedelta(seconds=cfg.orbit.mission_seconds))
    sgp_epoch = (epoch - datetime(1949, 12, 31, tzinfo=UTC)).total_seconds() / 86400
    mean_motion = math.sqrt(398600.8 / (6378.135 + cfg.orbit.altitude_km) ** 3) * 60
    count = cfg.training.clients
    base, remainder = divmod(count, cfg.orbit.planes)
    plane_sizes = [base + (plane < remainder) for plane in range(cfg.orbit.planes)]
    contacts, elements = [], []
    for client in range(count):
        offset = 0
        plane = 0
        phase = 0
        plane_size = 0
        for plane, plane_size in enumerate(plane_sizes):
            if client < offset + plane_size:
                phase = client - offset
                break
            offset += plane_size
        raan = 2 * math.pi * plane / cfg.orbit.planes
        anomaly = 2 * math.pi * (phase + plane / cfg.orbit.planes) / plane_size
        sat = Satrec()
        sat.sgp4init(
            WGS72,
            "i",
            90000 + client,
            sgp_epoch,
            0,
            0,
            0,
            0.0001,
            0,
            math.radians(cfg.orbit.inclination_deg),
            anomaly,
            mean_motion,
            raan,
        )
        satellite = EarthSatellite.from_satrec(sat, ts)
        if satellite.at(start).message:
            raise ValueError("SGP4 propagation failed at epoch")
        elements.append(
            {
                "client": client,
                "epoch": epoch.isoformat(),
                "raan_rad": raan,
                "mean_anomaly_rad": anomaly,
                "mean_motion_rad_min": mean_motion,
                "eccentricity": 0.0001,
                "bstar": 0,
                "inclination_deg": cfg.orbit.inclination_deg,
            }
        )
        for station in cfg.orbit.stations:
            site = wgs84.latlon(station.latitude, station.longitude, station.elevation_m)
            topocentric = satellite - site
            opened = (
                0.0 if topocentric.at(start).altaz()[0].degrees >= cfg.orbit.elevation_deg else None
            )
            times, events = satellite.find_events(
                site, start, end, altitude_degrees=cfg.orbit.elevation_deg
            )
            for time, event in zip(times, events):
                seconds = float((time - start) * 86400)
                if event == 0:
                    opened = max(0.0, seconds)
                elif event == 2 and opened is not None:
                    distance = float(
                        topocentric.at(
                            ts.from_datetime(epoch + timedelta(seconds=(opened + seconds) / 2))
                        )
                        .distance()
                        .km
                    )
                    contacts.append(
                        {
                            "client": client,
                            "station": station.name,
                            "start": opened,
                            "end": min(seconds, cfg.orbit.mission_seconds),
                            "delay_seconds": distance / 299792.458,
                        }
                    )
                    opened = None
            if opened is not None:
                contacts.append(
                    {
                        "client": client,
                        "station": station.name,
                        "start": opened,
                        "end": cfg.orbit.mission_seconds,
                        "delay_seconds": 0.05,
                    }
                )
    result = {
        "config_hash": cfg.fingerprint(),
        "interval": "[start,end)",
        "geometry": "Synthetic SGP4 mean elements, WGS72; altitude seeds mean motion only",
        "backbone": "Ideal terrestrial; no inter-satellite links",
        "elements": elements,
        "contacts": sorted(contacts, key=lambda c: (c["start"], c["client"], c["station"])),
    }
    path = project_path("artifacts/contacts") / (cfg.fingerprint() + ".json")
    atomic_json(path, result)
    return path
