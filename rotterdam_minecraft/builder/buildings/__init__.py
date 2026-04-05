from .euromast import build as build_euromast
from .witte_huis import build as build_witte_huis
from .de_rotterdam import build as build_de_rotterdam
from .depot_boijmans import build as build_depot_boijmans
from .erasmus_bridge import build as build_erasmus_bridge
from .markthal import build as build_markthal
from .cube_houses import build as build_cube_houses
from .rotterdam_centraal import build as build_rotterdam_centraal
from .hotel_new_york import build as build_hotel_new_york
from .van_nelle_factory import build as build_van_nelle_factory
from .heaven_temple import build as build_heaven_temple

ALL_BUILDINGS = {
    "euromast": build_euromast,
    "witte_huis": build_witte_huis,
    "de_rotterdam": build_de_rotterdam,
    "depot_boijmans": build_depot_boijmans,
    "erasmus_bridge": build_erasmus_bridge,
    "markthal": build_markthal,
    "cube_houses": build_cube_houses,
    "rotterdam_centraal": build_rotterdam_centraal,
    "hotel_new_york": build_hotel_new_york,
    "van_nelle_factory": build_van_nelle_factory,
    "heaven_temple": build_heaven_temple,
}
