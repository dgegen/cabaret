import pytest
from datetime import datetime, UTC
from cabaret.observatory import Observatory
from cabaret.camera import Camera
from cabaret.telescope import Telescope
from cabaret.site import Site


def test_observatory_initialization():
    observatory = Observatory()
    assert observatory.name == "Observatory"
    assert isinstance(observatory.camera, Camera)
    assert isinstance(observatory.telescope, Telescope)
    assert isinstance(observatory.site, Site)


def test_observatory_post_init():
    observatory = Observatory(
        camera={"name": "test_camera"},
        telescope={},
        site={},
    )
    assert isinstance(observatory.camera, Camera)
    assert isinstance(observatory.telescope, Telescope)
    assert isinstance(observatory.site, Site)


def test_generate_image():
    observatory = Observatory()
    dateobs = datetime.now(UTC)
    img = observatory.generate_image(
        ra=12.3323, dec=30.4343, exp_time=10, dateobs=dateobs, seed=0
    )
    assert img is not None


def test_to_dict():
    observatory = Observatory()
    obs_dict = observatory.to_dict()
    assert isinstance(obs_dict, dict)
    assert obs_dict["name"] == "Observatory"


def test_from_dict():
    config = {
        "camera": {"name": "test_camera"},
        "telescope": {},
        "site": {},
    }
    observatory = Observatory.from_dict(config)
    assert isinstance(observatory, Observatory)
    assert observatory.camera.name == "test_camera"


def test_save_to_yaml(tmp_path):
    observatory = Observatory()
    file_path = tmp_path / "observatory.yaml"
    observatory.save_to_yaml(file_path)
    assert file_path.exists()


def test_load_from_yaml(tmp_path):
    file_path = tmp_path / "observatory.yaml"
    Observatory(name="test", camera={"name": "test_camera"}).save_to_yaml(file_path)
    observatory = Observatory.load_from_yaml(file_path)
    assert isinstance(observatory, Observatory)
    assert observatory.name == "test"
    assert observatory.camera.name == "test_camera"


if __name__ == "__main__":
    pytest.main()
