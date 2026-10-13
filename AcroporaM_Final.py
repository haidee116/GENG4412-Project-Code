# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

import numpy as np
from opendrift.models.oceandrift import Lagrangian3DArray
from bottomdrifters import BottomDrifters
import logging
from opendrift.config import CONFIG_LEVEL_ESSENTIAL, CONFIG_LEVEL_BASIC, CONFIG_LEVEL_ADVANCED
from suntimes import SunTimes
import geopandas as gpd

logger = logging.getLogger(__name__)


class LarvalCoralElement(Lagrangian3DArray):
    """
    Extends Lagrangian3DArray with relevant properties for coral larvae.
    Each "element" is one virtual larva.
    """

    # variables: define extra fields stored for each larva
    variables = Lagrangian3DArray.add_variables([
        # Age since release in days (float)
        ('age_days', {'dtype': np.float32,
                      'units': 'days',
                      'default': 0.}),

        # Competency flag: 0 = not competent, 1 = competent to settle
        ('competent', {'dtype': np.uint8,
                       'units': '',
                       'default': 0}),

        # Settlement flag: 0 = not settled, 1 = settled
        ('settled', {'dtype': np.uint8,
                     'units': '',
                     'default': 0}),

        # Optional: ID of source reef (integer index, for connectivity)
        ('source_reef_id', {'dtype': np.int32,
                            'units': '',
                            'default': -1}),
    
        # ID of settlement reef (integer index, for connectivity)
        ('settlement_reef_id', {'dtype': np.int32,
                                'units': '',
                                'default': -1}),

        # Optional: ID of source point (integer index, for connectivity)
        ('source_point_id', {'dtype': np.int32,
                             'units': '',
                             'default': -1}),])


class LarvalCorals(BottomDrifters):

    # Tell OpenDrift which element type this model uses
    ElementType = LarvalCoralElement

    # Environmental variables required from forcing readers
    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'x_wind': {'fallback': 0},
        'y_wind': {'fallback': 0},
        'sea_surface_height': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 50},
        'ocean_vertical_diffusivity': {'fallback': 1e-5, 'profiles': True},
        'ocean_mixed_layer_thickness': {'fallback': 20},
        'sea_water_temperature': {'fallback': 10, 'profiles': True},
        'sea_water_salinity': {'fallback': 35, 'profiles': True},
        'upward_sea_water_velocity': {'fallback': 0}
    }

    def __init__(self, *args, **kwargs):
        
        # Call parent constructor (BottomDrifters) to initialise OpenDrift internals
        super(LarvalCorals, self).__init__(*args, **kwargs)

        #self.windage_factor = 0.03

        self._add_config({
            # Pre-competency duration (days)
            'biology:competency_start_days': {
                'type': 'float',
                'default': 4.0,
                'min': 0.0,
                'max': 30.0,
                'units': 'days',
                'description': 'Age at which larvae become competent to settle',
                'level': CONFIG_LEVEL_BASIC
            },
            # PLD (end of competency / deactivation)
            'biology:competency_end_days': {
                'type': 'float',
                'default': 30.0,
                'min': 0.0,
                'max': 120.0,
                'units': 'days',
                'description': 'Age after which larvae are no longer competent (PLD)',
                'level': CONFIG_LEVEL_BASIC
            },
            # Mortality rate per day
            'biology:mortality_rate_per_day': {
                'type': 'float',
                'default': 0.054,
                'min': 0.0,
                'max': 1.0,
                'units': '1/day',
                'description': 'Daily mortality probability',
                'level': CONFIG_LEVEL_ADVANCED
            },
            # Vertical behaviour scenario
            "biology:vertical_behavior_mode": {
                "type": "enum",
                "enum": ["S0", "S1", "S2", "S3"],
                "default": "S0",
                "description": "Vertical migration scenario.",
                "level": CONFIG_LEVEL_BASIC,
            },
            "biology:settlement_depth": {
                "type": "float",
                "default": 2.0,  # m
                "min": 0.0,
                "max": 10.0,
                "units": "m",
                "description": "Height above seabed at which larvae can settle",
                "level": CONFIG_LEVEL_BASIC,
            },
            "swim_speed": {
                "type": "float",
                "default": 0.002,  # m s-1
                "min": 0.0,
                "max": 0.1,
                "units": "m s-1",
                "description": "Active vertical swimming speed.",
                "level": CONFIG_LEVEL_BASIC,
            },
            "biology:max_depth": {
                "type": "float",
                "default": 100.0,
                "min": 0.0,
                "max": 6000.0,
                "units": "m",
                "description": "Larvae that reach this depth are killed (deactivated).",
                "level": CONFIG_LEVEL_ADVANCED,
            },
            "windage_factor": {
                "type": "float",
                 "default": 0.03,
                "min": 0.0,
                "max": 1.0,
                "units": "1",
                "description": "Windage factor.",
                "level": CONFIG_LEVEL_ADVANCED,
            },
            "windage_depth": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 50.0,
                "units": "m",
                "description": "Depth below the surface within which windage is applied.",
                "level": CONFIG_LEVEL_ADVANCED,
            }, 
        })
        
        self._set_config_default('general:seafloor_action', 'lift_to_seafloor')
        self._set_config_default("drift:wind_drift_depth", 0.5)

        self._mixing_by_mode = {
            "S0": False,  # forced to z=0 every step regardless; mixing would be wasted compute
            "S1": True,   # passive turbulent dispersal IS the scenario
            "S2": True,   # directed descent should still ride on ambient turbulence, not replace it
            "S3": True,   # DVM overlaid on turbulence, not instead of it
        }

        self._bathymetry_checked = False

    def update_larvae(self):
        dt_days = self.time_step.total_seconds() / 86400.0
        self.elements.age_days += dt_days

        comp_start = self.get_config('biology:competency_start_days')
        comp_end   = self.get_config('biology:competency_end_days')
        mort_rate  = self.get_config('biology:mortality_rate_per_day')

        n = len(self.elements.status)

        # --- Competency ---
        become_competent = (
            (self.elements.age_days >= comp_start) &
            (self.elements.age_days <  comp_end) &
            (self.elements.competent == 0) &
            (self.elements.status == 0)
        )
        self.elements.competent[become_competent] = 1

        # --- End of PLD ---
        expired_mask = (self.elements.age_days >= comp_end) & (self.elements.status == 0)
        if expired_mask.any():
            self.deactivate_elements(expired_mask, reason='post-PLD')

        # --- Mortality --- CHECK! maybe e to something?
        if mort_rate > 0:
            p = 1 - (1 - mort_rate) ** dt_days
            alive_idx = np.where(self.elements.status == 0)[0]
            if len(alive_idx) > 0:
                rand = np.random.rand(len(alive_idx))
                dead_mask = np.zeros(n, dtype=bool)
                dead_mask[alive_idx[rand < p]] = True
                if dead_mask.any():
                    self.deactivate_elements(dead_mask, reason='mortality')

    def day_or_night(self):
        # self.time is UTC; GBR uses AEST (UTC+10) year-round.
        local_hour = (self.time.hour + 10 + self.time.minute / 60) % 24

        return "day" if 5.0 <= local_hour < 19.0 else "night"
    
    
    def hold_precompetent_at_surface(self):
        """
        All scenarios: larvae that are not yet competent stay at the sea
        surface (z = 0). Scenario-specific vertical behaviour only starts once
        a larva has become competent.
        """
        active = np.where(self.elements.status == 0)[0]

        if len(active) == 0:
            return

        precompetent = active[self.elements.competent[active] == 0]

        if len(precompetent) > 0:
            self.elements.z[precompetent] = 0.0
            self.elements.terminal_velocity[precompetent] = 0.0

    def apply_vertical_behavior(self):
        """
        Apply the selected vertical-migration scenario.

        All scenarios: larvae are held at the surface until they are competent.
        Once competent:
            S0 - stay at the surface
            S1 - passive (drift with turbulent mixing)
            S2 - swim down to the benthic boundary layer
            S3 - diel vertical migration (descend by day, ascend by night)
        """
        active = np.where(self.elements.status == 0)[0]

        if len(active) == 0:
            return

        self.elements.terminal_velocity[active] = 0.0

        mode = self.get_config("biology:vertical_behavior_mode")
        if mode not in ("S0", "S1", "S2", "S3"):
            raise ValueError(f"Unknown vertical behaviour mode: {mode}")

        # Pre-competent larvae: surface, in every scenario
        self.hold_precompetent_at_surface()

        # Scenario behaviour applies to competent larvae only
        competent = active[self.elements.competent[active] == 1]

        if len(competent) == 0:
            return

        swim_speed = self.get_config("swim_speed")
        dt_seconds = self.time_step.total_seconds()

        if mode == "S0":
            self.elements.z[competent] = 0.0

        elif mode == "S1":
            pass

        elif mode == "S2":
            self.elements.terminal_velocity[competent] = -swim_speed

        elif mode == "S3":
            swim_velocity = -swim_speed if self.day_or_night() == "day" else swim_speed
            self.elements.terminal_velocity[competent] = swim_velocity

    def enforce_seafloor_boundary(self):

        active = self.elements.status == 0
    
        if not np.any(active):
            return
    
        active_indices = np.where(active)[0]
    
        # Get bathymetry
        seafloor_depth = self.environment.sea_floor_depth_below_sea_level[active]
        sea_surface_height = self.environment.sea_surface_height[active]
    
        # Convert bathymetry to OpenDrift z coordinate
        seafloor_z = -(seafloor_depth + sea_surface_height)
    
        # Current particle depths
        particle_z = self.elements.z[active]
    
        # Particles below seabed
        below_seafloor = particle_z < seafloor_z
    
        if np.any(below_seafloor):
    
            particles_below = active_indices[below_seafloor]
    
            # Move them exactly onto the seabed
            self.elements.z[particles_below] = seafloor_z[below_seafloor]
    
            logger.warning(
                "SEAFLOOR CORRECTION: %d particles moved to seabed | "
                "particle depth before correction: %.2f to %.2f m | "
                "seafloor depth: %.2f to %.2f m",
                len(particles_below),
                np.nanmin(particle_z[below_seafloor]),
                np.nanmax(particle_z[below_seafloor]),
                np.nanmin(seafloor_depth[below_seafloor]),
                np.nanmax(seafloor_depth[below_seafloor])
            )

    def enforce_depth_limit(self):
        """
        Deactivate ('kill') particles that have gone deeper than the
        configured maximum depth. This guards against unrealistic deep
        excursions — e.g. S2's seafloor-seeking behaviour swimming
        indefinitely downward over off-shelf water where the seabed is
        far deeper than any coral larva would realistically survive to.
        """
        active = np.where(self.elements.status == 0)[0]

        if len(active) == 0:
            return

        max_depth = self.get_config("biology:max_depth")
        too_deep = active[self.elements.z[active] <= -max_depth]

        if len(too_deep) > 0:
            mask = np.zeros(len(self.elements.status), dtype=bool)
            mask[too_deep] = True

            logger.warning(
                "DEPTH LIMIT: deactivating %d particles that reached %.1f m | "
                "min depth reached: %.2f m",
                len(too_deep), max_depth, np.nanmin(self.elements.z[too_deep])
            )

            self.deactivate_elements(mask, reason='exceeded_max_depth')

    def enforce_vertical_constraints(self):
        """
        Enforce biological vertical-depth constraints after all physical
        vertical transport has occurred.
        """

        active = np.where(self.elements.status == 0)[0]

        if len(active) == 0:
            return

        mode = self.get_config("biology:vertical_behavior_mode")

        # All scenarios: pre-competent larvae are put back at the surface
        # (undoes any displacement from mixing / vertical advection)
        self.hold_precompetent_at_surface()

        if mode == "S0":
            competent = active[self.elements.competent[active] == 1]
            self.elements.z[competent] = 0.0

        elif mode in ("S1", "S2", "S3"):
            pass

        else:
            raise ValueError(f"Unknown vertical behaviour mode: {mode}")

    def set_reef_polygons(self, reef_polygons, reef_id_column='GBR_ID', reef_name_column='GBR_NAME'):
        """
        Store reef polygons used for settlement.
    
        reef_polygons should be a GeoDataFrame in lon/lat or convertible to EPSG:4326.
        """
        if reef_polygons.crs is None:
            reef_polygons = reef_polygons.set_crs("EPSG:4326")
        else:
            reef_polygons = reef_polygons.to_crs("EPSG:4326")
    
        self.reef_polygons = reef_polygons.copy()
        self.reef_id_column = reef_id_column
        self.reef_name_column = reef_name_column
    
    
    def larvae_over_reef(self, idx):
        """
        For larval element indices, return which are over reef and which reef they hit.
    
        Uses polygon intersection, so a larva settles if its current lon/lat lies inside
        or on the edge of a reef polygon.
        """
        if not hasattr(self, 'reef_polygons'):
            raise ValueError("No reef polygons set. Call o.set_reef_polygons(reefs) before o.run().")
    
        points = gpd.GeoDataFrame(
            {'element_index': idx},
            geometry=gpd.points_from_xy(self.elements.lon[idx], self.elements.lat[idx]),
            crs="EPSG:4326"
        )
    
        keep_cols = [self.reef_id_column, self.reef_name_column, 'geometry']
        reef_polygons = self.reef_polygons[keep_cols]
    
        joined = gpd.sjoin(
            points,
            reef_polygons,
            how='left',
            predicate='intersects'
        )
    
        joined = joined.dropna(subset=['index_right'])
    
        over_reef = np.zeros(len(idx), dtype=bool)
        reef_ids = np.full(len(idx), -1, dtype=np.int32)
    
        if len(joined) == 0:
            return over_reef, reef_ids
    
        # If a point intersects more than one reef polygon, keep the first match.
        joined = joined.groupby(joined.index).first()
    
        matched_positions = joined.index.to_numpy()
        over_reef[matched_positions] = True
        reef_ids[matched_positions] = joined[self.reef_id_column].astype(int).to_numpy()
    
        return over_reef, reef_ids
    


    def check_settlement(self):
        """
        Settle competent larvae only when they are over a reef and within the
        configured benthic boundary layer above the local seabed.
        """
        idx = np.where(
            (self.elements.status == 0) &
            (self.elements.competent == 1) &
            (self.elements.settled == 0)
        )[0]
    
        if len(idx) == 0:
            return
    
        over_reef, reef_ids = self.larvae_over_reef(idx)

        mode = self.get_config("biology:vertical_behavior_mode")

        if mode == "S0" or mode == "S1":
            # Settle competent larvae that are over a reef, ignoring the BBL
            settle_here = over_reef
        else:

            # OpenDrift z is positive upward, relative to the instantaneous sea surface.
            # The seafloor's z-coordinate is therefore negative.
            seafloor_z = -(
                self.environment.sea_floor_depth_below_sea_level[idx]
                + self.environment.sea_surface_height[idx]
            )
        
            height_above_seabed = self.elements.z[idx] - seafloor_z
            bbl_height = self.get_config("biology:settlement_depth")
        
            # The lower bound prevents settlement below the modelled seabed.
            within_bbl = (
                (height_above_seabed >= 0) &
                (height_above_seabed <= bbl_height)
            )
        
            settle_here = over_reef & within_bbl
        settle_idx = idx[settle_here]
    
        if len(settle_idx) > 0:
            self.elements.settled[settle_idx] = 1
            self.elements.settlement_reef_id[settle_idx] = reef_ids[settle_here]
    
            settle_mask = np.zeros(len(self.elements.status), dtype=bool)
            settle_mask[settle_idx] = True
            self.deactivate_elements(settle_mask, reason="settled")

    def check_bathymetry(self):

        active = (self.elements.status == 0)

        if not np.any(active):
            return

        depth = (self.environment.sea_floor_depth_below_sea_level[active])

        logger.info(
            "Seafloor depth seen by active particles: min=%.2f max=%.2f mean=%.2f",
            np.nanmin(depth), np.nanmax(depth), np.nanmean(depth)
        )
        
        if np.allclose(
            depth,
            50.0
        ):

            logger.warning(
                "WARNING: All active particles are seeing a seafloor depth of approximately 50 m. This strongly suggests the bathymetry reader is not providing 'sea_floor_depth_below_sea_level' and OpenDrift is using the 50 m fallback."
            )

    
    def update(self):
        if not self._bathymetry_checked:
            self.check_bathymetry()
            self._bathymetry_checked = True
    
        # 1. Update biological state
        self.update_larvae()

        # 1b. Pre-competent larvae sit at the surface in every scenario
        #     (also moves freshly seeded larvae from their release depth before transport)
        self.hold_precompetent_at_surface()
    
        # 2. Horizontal transport
        self.advect_ocean_current()

        # 3. Horizontal windage
        # Apply 3% windage only to particles in the upper 0.5 m

        active = np.where(self.elements.status == 0)[0]

        if len(active) > 0:

            windage_coef  = self.get_config("windage_factor")
            windage_depth = self.get_config("windage_depth")

            # OpenDrift z:
            #   0 m = sea surface
            #  -windage_depth m = windage_depth m below surface
            surface = (
                (self.elements.z[active] >= -windage_depth) &
                (self.elements.z[active] <= 0.0)
            )

            # No windage below windage_depth
            self.elements.wind_drift_factor[active] = 0.0

            # Apply configured windage coefficient in the surface layer
            self.elements.wind_drift_factor[active[surface]] = windage_coef

            self.advect_wind()
    
        # 4. Physical terminal velocity (density/diameter based — likely near-zero for larvae)
        self.update_terminal_velocity()
        
        # 5. Biological vertical behaviour overrides terminal velocity for this step
        self.apply_vertical_behavior()
    
        # 6. Apply vertical mixing / buoyancy — now uses the swim velocity just set
        mode = self.get_config("biology:vertical_behavior_mode")
        if self._mixing_by_mode.get(mode, True):
            self.vertical_mixing()
        else:
            self.vertical_buoyancy()
    
         # 7. Apply vertical advection
        self.vertical_advection()

        # 7b. Kill particles that have gone unrealistically deep
        self.enforce_depth_limit()

        # 8. Enforce biological depth constraints
        self.enforce_vertical_constraints()
    
        # 9. HARD SEAFLOOR BOUNDARY
        self.enforce_seafloor_boundary()
    
        # 10. Check settlement
        self.check_settlement()