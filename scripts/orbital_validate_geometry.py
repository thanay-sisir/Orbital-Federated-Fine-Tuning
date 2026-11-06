import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, CartesianRepresentation, EarthLocation, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import Satrec, WGS72
from skyfield.api import EarthSatellite, load, wgs84


ROOT = Path(__file__).resolve().parents[1]


def crossings(elevation, mission, spacing, mask, tolerance=0.01):
    grid = np.arange(0, mission + spacing, spacing)
    grid[-1] = mission
    visible = elevation(grid) >= mask
    opened = 0.0 if visible[0] else None
    intervals = []
    for index in np.flatnonzero(visible[:-1] != visible[1:]):
        left, right = grid[index], grid[index + 1]
        left_visible = bool(visible[index])
        while right - left > tolerance:
            middle = (left + right) / 2
            if bool(elevation(np.array([middle]))[0] >= mask) == left_visible:
                left = middle
            else:
                right = middle
        event = (left + right) / 2
        if not left_visible:
            opened = event
        else:
            intervals.append([opened, event])
            opened = None
    if opened is not None:
        intervals.append([opened, mission])
    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/orbital/mission-repeat-v1-s42-fixed")
    parser.add_argument("--output", default="results/orbital/engineering-v1/geometry.json")
    args = parser.parse_args()
    run = ROOT / args.run
    archived = json.loads((run / "contacts.json").read_text())
    cfg = json.loads((run / "manifest.json").read_text())["config"]
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    timescale = load.timescale(builtin=True)
    epoch = datetime.fromisoformat(cfg["orbit"]["epoch"]).astimezone(UTC)
    epoch_time = Time(epoch, scale="utc")
    sgp_epoch = (epoch - datetime(1949, 12, 31, tzinfo=UTC)).total_seconds() / 86400
    mission = cfg["orbit"]["mission_seconds"]
    mask = cfg["orbit"]["elevation_deg"]
    checks, samples = [], []
    for element in archived["elements"]:
        client = element["client"]
        sat = Satrec()
        sat.sgp4init(WGS72, "i", 90000 + client, sgp_epoch, element["bstar"], 0, 0,
                     element["eccentricity"], 0, math.radians(element["inclination_deg"]),
                     element["mean_anomaly_rad"], element["mean_motion_rad_min"], element["raan_rad"])
        satellite = EarthSatellite.from_satrec(sat, timescale)
        for station in cfg["orbit"]["stations"]:
            location = EarthLocation.from_geodetic(station["longitude"] * u.deg, station["latitude"] * u.deg, station["elevation_m"] * u.m)
            sf_site = wgs84.latlon(station["latitude"], station["longitude"], station["elevation_m"])

            def astropy_elevation(seconds):
                time = epoch_time + np.atleast_1d(seconds) * u.s
                error, position, velocity = sat.sgp4_array(time.jd1, time.jd2)
                if np.any(error):
                    raise ValueError("Independent SGP4 propagation error")
                teme = TEME(CartesianRepresentation(position.T * u.km), obstime=time)
                geocentric = teme.transform_to(ITRS(obstime=time))
                relative = geocentric.cartesian.without_differentials() - location.get_itrs(time).cartesian
                topocentric = ITRS(relative, obstime=time, location=location)
                return topocentric.transform_to(AltAz(obstime=time, location=location, pressure=0 * u.hPa)).alt.deg

            intervals = crossings(astropy_elevation, mission, 10, mask)
            fine = crossings(astropy_elevation, mission, 5, mask)
            expected = [[c["start"], c["end"]] for c in archived["contacts"] if c["client"] == client and c["station"] == station["name"]]
            expected.sort()
            if len(intervals) != len(expected) or len(fine) != len(expected):
                raise AssertionError("Independent event search disagrees on contact count")
            event_error = np.asarray(intervals) - np.asarray(expected)
            convergence = float(np.max(np.abs(np.asarray(fine) - intervals))) if intervals else 0
            sample_times = np.arange(0, mission + 1, 60)
            astro_alt = astropy_elevation(sample_times)
            sf_times = timescale.from_astropy(epoch_time + sample_times * u.s)
            sf_alt = (satellite - sf_site).at(sf_times).altaz()[0].degrees
            differences = astro_alt - sf_alt
            maximum = float(np.max(np.abs(differences)))
            edge_maximum = float(np.max(np.abs(event_error))) if intervals else 0
            passed = maximum <= 0.02 and edge_maximum <= 2 and convergence <= 0.02
            checks.append({"client": client, "station": station["name"], "contacts": len(expected),
                           "max_elevation_difference_deg": maximum, "max_boundary_difference_seconds": edge_maximum,
                           "grid_convergence_seconds": convergence, "boundary_deltas_seconds": event_error.tolist(),
                           "reference_intervals": intervals, "passed": passed})
            for seconds, a, s in zip(sample_times, astro_alt, sf_alt):
                samples.append({"client": client, "station": station["name"], "seconds": int(seconds),
                                "astropy_elevation_deg": float(a), "skyfield_elevation_deg": float(s)})
    result = {"status": "passed" if all(c["passed"] for c in checks) else "failed", "run": args.run,
              "contact_file_sha256": hashlib.sha256((run / "contacts.json").read_bytes()).hexdigest(),
              "method": "Shared python-sgp4 propagator and archived mean elements; independent Astropy TEME/ITRS/topocentric transforms and grid/bisection event search versus Skyfield events.",
              "thresholds": {"elevation_deg": 0.02, "boundary_seconds": 2, "grid_convergence_seconds": 0.02},
              "grids_seconds": [10, 5], "bisection_tolerance_seconds": 0.01,
              "earth_orientation": "Offline bundled Astropy IERS and Skyfield builtin timescale; no EOP downloads or refraction.",
              "versions": {p: version(p) for p in ("astropy", "astropy-iers-data", "skyfield", "sgp4", "numpy")},
              "checks": checks, "samples": samples,
              "limitations": ["Numerical cross-check of synthetic geometry, not validation against measured spacecraft passes.",
                              "SGP4 is shared by both methods; this does not independently validate the force model.",
                              "Two coarse grids plus bisection do not prove absence of sub-five-second grazing passes.",
                              "Radio link budgets, weather, antenna steering, energy and spacecraft compute remain uncalibrated."]}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "checks")}, indent=2))
    if result["status"] != "passed":
        raise SystemExit("Geometry cross-check exceeded predeclared tolerances")


if __name__ == "__main__":
    main()
