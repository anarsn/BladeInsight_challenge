import unittest
import raccoon_processer

class TestRaccoonProcesser(unittest.TestCase):
 def test_normalize_cords(self):
        self.assertEqual(raccoon_processer.normalize_cords(110,119,167,217,199,253), (0.13316582914572864,0.041501976284584984,0.9723618090452262, 0.8992094861660079))

 def test_check_for_intersections(self):
        self.assertEqual(raccoon_processer.check_for_intersections(   
            [
                {
                    "xmin": 0.15605095541401273,
                    "ymin": 0.2786624203821656,
                    "xmax": 0.5955414012738853,
                    "ymax": 0.9633757961783439,
                    "label": "Raccoon",
                    "width_to_height": 0.6418604651162791
                },
                {
                    "xmin": 0.2754777070063694,
                    "ymin": 0.0031847133757961785,
                    "xmax": 0.75,
                    "ymax": 0.9840764331210191,
                    "label": "Raccoon",
                    "width_to_height": 0.4837662337662338
                }
               
            ] ,628,314),43214.99999999999)
        self.assertEqual(raccoon_processer.check_for_intersections(   
           [
                {
                    "xmin": 0.13316582914572864,
                    "ymin": 0.041501976284584984,
                    "xmax": 0.9723618090452262,
                    "ymax": 0.8992094861660079,
                    "label": "Raccoon",
                    "width_to_height": 0.9784174327860501
                }
            ] ,253,199),0)


if __name__ == "__main__":
        unittest.main() 