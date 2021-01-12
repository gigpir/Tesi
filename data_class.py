class Song:
    def __init__(self, id, name, hotttnesss):
        self.name = name
        self.id = id
        self.hotttnesss = hotttnesss
        self.id_artist=''
        self.bars_start = None
        self.bars_confidence = None
        self.beats_start = None
        self.beats_confidence = None
        self.danceability = None
        self.duration = None
        self.end_of_fade_in = None
        self.energy = None
        self.key = None
        self.loudness = None
        self.mode = None
        self.mode_confidence = None
        self.start_of_fade_out = None
        self.tempo = None
        self.time_signature = None
        self.time_signature_confidence = None
        self.track_id = None
        self.segments_start = None
        self.segments_confidence = None
        self.segments_pitches = None
        self.segments_timbre = None
        self.segments_loudness_max = None
        self.segments_loudness_max_time = None
        self.segments_loudness_start = None
        self.sections_start = None
        self.sections_confidence = None
        self.tatums_start = None
        self.tatums_confidence = None
    def __str__(self):
        return str(self.name)
class Artist:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.terms = []
        self.terms_freq = []
        self.terms_weight = []
        self.similar_artists = [] #Ground truth?
        self.song_list = []
        self.has_N_similar_artists = True
    def has_zero_terms(self):
        return len(self.terms) == 0
    def get_terms_num(self):
        return len(self.terms)