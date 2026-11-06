"""
Automated Test Suite for Smart Notes Assistant
Tests:
- Adding notes
- Editing notes
- Deleting notes
- Saving and retrieving notes
- Model training
- Category predictions
- Uncertainty detection
- Manual category correction
- Keyword search
- Similarity search
- Category filtering
- Duplicate prevention
"""

import os
import sys
import unittest
import tempfile
import shutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classifier import NoteClassifier, CATEGORIES, preprocess_text
from storage import NotesStorage
from search import NoteSearchEngine


class TestSmartNotesAssistant(unittest.TestCase):
    def setUp(self):
        """Set up temporary test database and clean classifier."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_notes.db")
        self.storage = NotesStorage(self.db_path)

        # Train a fast in-memory classifier for testing
        self.classifier = NoteClassifier()
        sample_texts = [
            "My Python exam is on Friday at 10 AM",
            "Submit machine learning homework",
            "Prepare quarterly presentation for Monday client meeting",
            "Sprint review with engineering team",
            "Pay the electricity bill before the 28th",
            "Transfer monthly rent money",
            "Flight to Delhi is booked for Monday morning",
            "Reserve hotel room in Tokyo",
            "Buy groceries this weekend",
            "Pick up milk and bread",
            "Dentist checkup appointment on Thursday 4 PM",
            "Take daily vitamins and medication",
            "Alex's birthday party on Saturday evening",
            "Attend live concert tomorrow night",
            "Meditate for 15 minutes before sleeping",
            "Journal thoughts and gratitude",
        ]
        sample_labels = [
            "Education",
            "Education",
            "Work",
            "Work",
            "Finance",
            "Finance",
            "Travel",
            "Travel",
            "Shopping",
            "Shopping",
            "Health",
            "Health",
            "Events",
            "Events",
            "Personal",
            "Personal",
        ]
        self.classifier.fit(sample_texts, sample_labels, max_iter=100)

    def tearDown(self):
        """Clean up temporary test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Test Saving & Retrieving Notes
    def test_save_and_retrieve_note(self):
        note_id = self.storage.add_note(
            text="Buy organic vegetables and fruits",
            category="Shopping",
            confidence=0.92,
            is_uncertain=False,
        )
        self.assertGreater(note_id, 0)

        retrieved = self.storage.get_note_by_id(note_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["text"], "Buy organic vegetables and fruits")
        self.assertEqual(retrieved["category"], "Shopping")
        self.assertAlmostEqual(retrieved["confidence"], 0.92, places=2)
        self.assertFalse(retrieved["is_uncertain"])

    # 2. Test Adding Notes
    def test_add_multiple_notes(self):
        initial_count = self.storage.get_note_count()
        self.assertEqual(initial_count, 0)

        self.storage.add_note("Note 1", "Work")
        self.storage.add_note("Note 2", "Finance")
        self.storage.add_note("Note 3", "Education")

        self.assertEqual(self.storage.get_note_count(), 3)
        all_notes = self.storage.get_all_notes()
        self.assertEqual(len(all_notes), 3)

    # 3. Test Editing Notes
    def test_edit_note(self):
        note_id = self.storage.add_note("Dentist appointment on Tuesday", "Health")
        success = self.storage.update_note(
            note_id=note_id,
            text="Dentist appointment rescheduled to Friday 3 PM",
            category="Health",
            confidence=0.98,
        )
        self.assertTrue(success)

        updated = self.storage.get_note_by_id(note_id)
        self.assertEqual(updated["text"], "Dentist appointment rescheduled to Friday 3 PM")
        self.assertAlmostEqual(updated["confidence"], 0.98, places=2)

    # 4. Test Deleting Notes
    def test_delete_note(self):
        note_id = self.storage.add_note("Temporary reminder", "Personal")
        self.assertEqual(self.storage.get_note_count(), 1)

        deleted = self.storage.delete_note(note_id)
        self.assertTrue(deleted)
        self.assertEqual(self.storage.get_note_count(), 0)
        self.assertIsNone(self.storage.get_note_by_id(note_id))

    # 5. Test Category Filtering
    def test_category_filtering(self):
        self.storage.add_note("Pay electricity bill", "Finance")
        self.storage.add_note("Transfer rent", "Finance")
        self.storage.add_note("Python exam review", "Education")

        finance_notes = self.storage.get_all_notes(category_filter="Finance")
        self.assertEqual(len(finance_notes), 2)
        for n in finance_notes:
            self.assertEqual(n["category"], "Finance")

        education_notes = self.storage.get_all_notes(category_filter="Education")
        self.assertEqual(len(education_notes), 1)
        self.assertEqual(education_notes[0]["category"], "Education")

    # 6. Test Duplicate Prevention
    def test_duplicate_prevention(self):
        samples = [
            {"text": "Flight to Delhi is booked for Monday morning.", "category": "Travel"},
            {"text": "Flight to Delhi is booked for Monday morning.", "category": "Travel"},
            {"text": "flight to delhi is booked for monday morning. ", "category": "Travel"},
            {"text": "Unique new note", "category": "Work"},
        ]
        added = self.storage.load_sample_notes(samples)
        self.assertEqual(added, 2)  # Only 2 distinct notes should be added

        # Second load attempt should add 0 duplicates
        added_second = self.storage.load_sample_notes(samples)
        self.assertEqual(added_second, 0)
        self.assertEqual(self.storage.get_note_count(), 2)

    # 7. Test Model Training & Evaluation
    def test_model_training_and_evaluation(self):
        self.assertTrue(self.classifier.is_trained)
        self.assertGreater(len(self.classifier.vocabulary), 0)

        # Test evaluation method
        test_texts = [
            "Submit machine learning assignment",
            "Pay monthly utility bill",
        ]
        test_labels = ["Education", "Finance"]
        metrics = self.classifier.evaluate(test_texts, test_labels)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)

    # 8. Test Category Predictions
    def test_category_predictions(self):
        pred_edu = self.classifier.predict("Study for machine learning exam")
        self.assertIn(pred_edu["category"], CATEGORIES)
        self.assertGreaterEqual(pred_edu["confidence"], 0.0)
        self.assertLessEqual(pred_edu["confidence"], 1.0)

        pred_fin = self.classifier.predict("Pay the credit card bill")
        self.assertIn(pred_fin["category"], CATEGORIES)

    # 9. Test Uncertainty Detection
    def test_uncertainty_detection(self):
        # Extremely ambiguous or empty text should trigger uncertainty
        ambiguous_res = self.classifier.predict("random gibberish xyz abc 12345")
        self.assertTrue(ambiguous_res["is_uncertain"])
        self.assertIn("Not sure about this note's category", ambiguous_res["uncertainty_message"])

    # 10. Test Manual Category Correction
    def test_manual_category_correction(self):
        # Add note with auto prediction
        pred = self.classifier.predict("Review notes")
        note_id = self.storage.add_note(
            text="Review notes",
            category=pred["category"],
            confidence=pred["confidence"],
            is_uncertain=pred["is_uncertain"],
        )

        # User manually overrides to "Work"
        self.storage.update_note(note_id, category="Work", confidence=1.0, is_uncertain=False)
        corrected = self.storage.get_note_by_id(note_id)
        self.assertEqual(corrected["category"], "Work")
        self.assertEqual(corrected["confidence"], 1.0)
        self.assertFalse(corrected["is_uncertain"])

    # 11. Test Keyword Search
    def test_keyword_search(self):
        notes = [
            {"id": 1, "text": "Pay the electricity bill before the 28th.", "category": "Finance"},
            {"id": 2, "text": "Flight to Delhi is booked for Monday.", "category": "Travel"},
            {"id": 3, "text": "Buy groceries this weekend.", "category": "Shopping"},
        ]
        engine = NoteSearchEngine(notes)
        results = engine.keyword_search("electricity")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], 1)
        self.assertGreater(results[0]["relevance_score"], 0.0)

    # 12. Test Similarity Search
    def test_similarity_search(self):
        notes = [
            {"id": 1, "text": "Pay the electricity bill before the 28th.", "category": "Finance"},
            {"id": 2, "text": "Flight to Delhi is booked for Monday.", "category": "Travel"},
            {"id": 3, "text": "Buy groceries this weekend.", "category": "Shopping"},
        ]
        engine = NoteSearchEngine(notes)
        # Natural language query "electricity payment" should match electricity bill
        results = engine.similarity_search("electricity payment")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], 1)
        self.assertGreater(results[0]["relevance_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
