"""Paired comparisons must not reward more requests or undetected source areas."""
import unittest
from aligned_prompt_audit import eligible, matched_table


def row(prompt, track, score, visible=1, status='measured'):
    return {'prompt_id': prompt, 'model': 'test/model', 'track': track,
            'settings': {'quality': 'high'},
            'alignment': {'status': status, 'visible_source_fraction': visible},
            'aligned': {'ssim_luminance': score, 'pixels_within_8_rgb_fraction': score,
                        'mae_rgb_0_255': 1-score, 'worst_tile_ssim': score}}


class MatchedComparisonTests(unittest.TestCase):
    def test_equal_weight_per_cover_despite_more_repeats(self):
        rows = [row('a', 'one', .9)] * 10 + [row('a', 'two', .1)]
        table = matched_table(rows, [('test/model', 'one'), ('test/model', 'two')], ['a'])
        self.assertAlmostEqual(table[0]['mean_ssim'], .5)

    def test_missing_condition_does_not_receive_mean(self):
        table = matched_table([row('a', 'one', .99)],
                              [('test/model', 'one'), ('test/model', 'two')], ['a'])
        self.assertFalse(table[0]['complete'])
        self.assertIsNone(table[0]['mean_ssim'])

    def test_partial_or_unavailable_alignment_cannot_win(self):
        self.assertFalse(eligible(row('a', 'one', 1, visible=.9)))
        self.assertFalse(eligible(row('a', 'one', 1, status='unavailable')))
        rows = [row('a', 'one', 1, visible=.9), row('b', 'one', .8)]
        table = matched_table(rows, [('test/model', 'one')], ['a', 'b'])
        self.assertEqual(table[0]['prompt_id'], 'b')
        self.assertIsNone(table[1]['mean_ssim'])


if __name__ == '__main__':
    unittest.main()
