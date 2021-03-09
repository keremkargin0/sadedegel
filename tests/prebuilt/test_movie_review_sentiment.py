from .context import movie_reviews

def test_model_load():
    model = movie_reviews.load()

    pred = model.predict(['çok sıkıcı'])

    assert pred[0] == 0

    pred = model.predict(['kesinlikle tavsiye ederim'])

    assert pred[0] == 1





