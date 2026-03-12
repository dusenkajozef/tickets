from tickets.data import load_data


class TestDataLoading():
    def test_load_data(self):
        data = load_data()
        assert "train" in data
        assert "test" in data
        assert len(data["train"]) > 0
        assert len(data["test"]) > 0
