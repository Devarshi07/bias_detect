# hotel_bias_detection.py

import os
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Dict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HotelBiasDetection:
    """Bias detection for hotel chatbot responses (no embeddings version)"""
    
    def __init__(
        self, 
        reviews_path: str,
        responses_parquet_path: str,
        output_dir: str = "evaluation/results"
    ):
        """
        Initialize bias detection
        
        Args:
            reviews_path: Path to hotel reviews CSV
            responses_parquet_path: Path to chatbot responses parquet file
            output_dir: Directory to save results
        """
        self.reviews_path = reviews_path
        self.responses_parquet_path = responses_parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.detailed_results = []
        
        logger.info("Initializing bias detection (keyword-based version)...")
        
        # Sparse data keywords (no embeddings needed!)
        self.sparse_keywords = [
            "limited review", "few review", "not enough", "small sample",
            "based on limited", "insufficient", "newly opened", "not many",
            "handful of review", "limited data", "only a few", "limited feedback",
            "sparse data", "not much feedback", "minimal reviews"
        ]
        
        # Configuration
        self.config = {
            'neg_sentiment_threshold': 0.7,
            'min_reviews_threshold': 4,
            'rating_disparity_threshold': 4.0
        }
        
        logger.info("✓ Bias detection initialized")
    
    
    def load_reviews(self) -> pd.DataFrame:
        """Load hotel reviews from CSV"""
        logger.info(f"Loading reviews from: {self.reviews_path}")
        
        try:
            reviews_df = pd.read_csv(self.reviews_path)
            logger.info(f"✓ Loaded {len(reviews_df)} reviews")
            logger.info(f"✓ Columns: {list(reviews_df.columns)}")
            logger.info(f"✓ Unique hotels: {reviews_df['hotel_id'].nunique()}")
            
            return reviews_df
        except Exception as e:
            logger.error(f"Failed to load reviews: {e}")
            raise
    
    
    def load_chatbot_responses(self) -> pd.DataFrame:
        """Load chatbot responses from parquet file"""
        logger.info(f"Loading chatbot responses from: {self.responses_parquet_path}")
        
        if not os.path.exists(self.responses_parquet_path):
            logger.error(f"Chatbot responses file not found: {self.responses_parquet_path}")
            raise FileNotFoundError(f"Responses parquet not found at {self.responses_parquet_path}")
        
        try:
            responses_df = pd.read_parquet(self.responses_parquet_path)
            logger.info(f"✓ Loaded {len(responses_df)} chatbot responses")
            logger.info(f"✓ Columns: {list(responses_df.columns)}")
            logger.info(f"✓ Unique hotels: {responses_df['hotel_id'].nunique()}")
            
            # Validate required columns
            required_cols = ['hotel_id', 'answer']
            missing_cols = [col for col in required_cols if col not in responses_df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Required columns: {required_cols}")
                logger.info(f"Available columns: {list(responses_df.columns)}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return responses_df
            
        except Exception as e:
            logger.error(f"Failed to load chatbot responses: {e}")
            raise
    
    
    def simple_sentiment_analysis(self, text: str, rating: float = None) -> str:
        """Rule-based sentiment analysis"""
        
        # If rating is provided, use it
        if rating is not None:
            if rating >= 4:
                return 'positive'
            elif rating <= 2:
                return 'negative'
            else:
                return 'neutral'
        
        # Otherwise, use text analysis
        text_lower = text.lower()
        
        positive_words = [
            'great', 'excellent', 'amazing', 'good', 'recommend', 'love', 
            'best', 'wonderful', 'perfect', 'outstanding', 'fantastic',
            'enjoyed', 'comfortable', 'clean', 'friendly', 'helpful',
            'nice', 'pleasant', 'satisfied', 'happy'
        ]
        negative_words = [
            'bad', 'terrible', 'awful', 'poor', 'worst', 'hate', 
            'disappointing', 'horrible', 'dirty', 'rude', 'noisy',
            'outdated', 'uncomfortable', 'expensive', 'unpleasant',
            'issues', 'problems', 'complaints', 'dissatisfied'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    
    def analyze_review_sentiments(self, reviews_df: pd.DataFrame) -> Counter:
        """Analyze sentiments of all reviews"""
        sentiment_counts = Counter({"positive": 0, "neutral": 0, "negative": 0})
        
        for _, row in reviews_df.iterrows():
            sentiment = self.simple_sentiment_analysis(
                row['review_text'], 
                row['overall_rating']
            )
            sentiment_counts[sentiment] += 1
        
        return sentiment_counts
    
    
    def analyze_response_sentiment(self, response: str) -> Dict[str, float]:
        """Get sentiment probabilities for chatbot response"""
        sentiment = self.simple_sentiment_analysis(response)
        
        if sentiment == 'positive':
            return {"positive": 0.8, "neutral": 0.15, "negative": 0.05}
        elif sentiment == 'negative':
            return {"positive": 0.05, "neutral": 0.15, "negative": 0.8}
        else:
            return {"positive": 0.3, "neutral": 0.5, "negative": 0.2}
    
    
    def check_sparse_data_acknowledgment(self, response: str) -> bool:
        """Check if response acknowledges limited data (keyword-based)"""
        response_lower = response.lower()
        
        # Check if any sparse data keyword is mentioned
        for keyword in self.sparse_keywords:
            if keyword in response_lower:
                logger.debug(f"  Found sparse data acknowledgment: '{keyword}'")
                return True
        
        return False
    
    
    def detect_bias(
        self, 
        hotel_id: str,
        response: str, 
        reviews_df: pd.DataFrame
    ) -> Dict:
        """Main bias detection logic"""
        
        num_reviews = len(reviews_df)
        bias_flags = {
            "hotel_id": hotel_id,
            "bias_detected": False, 
            "bias_types": [],
            "num_reviews": num_reviews,
            "avg_rating": 0.0,
            "review_sentiments": {},
            "response_sentiment": {},
            "details": {}
        }
        
        if reviews_df.empty:
            logger.warning(f"No reviews for hotel {hotel_id}")
            return bias_flags
        
        # Calculate metrics
        avg_rating = reviews_df['overall_rating'].mean()
        bias_flags["avg_rating"] = float(avg_rating)
        
        # Analyze sentiments
        review_sentiments = self.analyze_review_sentiments(reviews_df)
        response_sentiment = self.analyze_response_sentiment(response)
        
        bias_flags["review_sentiments"] = dict(review_sentiments)
        bias_flags["response_sentiment"] = response_sentiment
        
        review_pos = review_sentiments['positive']
        review_neg = review_sentiments['negative']
        response_neg = response_sentiment['negative']
        
        # Bias 1: Over-reliance on negative reviews
        if response_neg > self.config['neg_sentiment_threshold'] and review_neg > review_pos:
            bias_flags["bias_detected"] = True
            bias_flags["bias_types"].append("over_reliance_on_negative")
            bias_flags["details"]["over_reliance_on_negative"] = {
                "response_negativity": response_neg,
                "negative_reviews": review_neg,
                "positive_reviews": review_pos
            }
            logger.info(f"  ⚠️  Detected: Over-reliance on negative reviews")
        
        # Bias 2: Missing data acknowledgment
        if num_reviews < self.config['min_reviews_threshold']:
            acknowledged = self.check_sparse_data_acknowledgment(response)
            if not acknowledged:
                bias_flags["bias_detected"] = True
                bias_flags["bias_types"].append("missing_data_acknowledgment")
                bias_flags["details"]["missing_data_acknowledgment"] = {
                    "num_reviews": num_reviews,
                    "acknowledged": acknowledged
                }
                logger.info(f"  ⚠️  Detected: Missing data acknowledgment")
        
        # Bias 3: Rating disparity (negative response despite good ratings)
        if (response_neg > self.config['neg_sentiment_threshold'] and 
            avg_rating >= self.config['rating_disparity_threshold']):
            bias_flags["bias_detected"] = True
            bias_flags["bias_types"].append("rating_disparity_bias")
            bias_flags["details"]["rating_disparity_bias"] = {
                "avg_rating": avg_rating,
                "response_negativity": response_neg
            }
            logger.info(f"  ⚠️  Detected: Rating disparity bias")
        
        if not bias_flags["bias_detected"]:
            logger.info(f"  ✓ No bias detected")
        
        return bias_flags
    
    
    def save_results(self):
        """Save results in JSON, Parquet, and CSV formats"""
        
        # Save detailed JSON results
        json_path = self.output_dir / "hotel-bias-scores.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"✓ Saved JSON results: {json_path}")
        
        # Save detailed Parquet results
        if self.detailed_results:
            detailed_df = pd.DataFrame(self.detailed_results)
            
            parquet_path = self.output_dir / "hotel-bias-results.parquet"
            detailed_df.to_parquet(parquet_path, index=False)
            logger.info(f"✓ Saved Parquet results: {parquet_path}")
            
            # Also save as CSV for easy viewing
            csv_path = self.output_dir / "hotel-bias-results.csv"
            detailed_df.to_csv(csv_path, index=False)
            logger.info(f"✓ Saved CSV results: {csv_path}")
    
    
    def generate_summary_report(self):
        """Generate and save summary report"""
        
        total_hotels = len(self.results)
        biased_hotels = sum(1 for v in self.results.values() if v.get("bias_detected", False))
        
        # Count bias types
        bias_type_counts = Counter()
        for result in self.results.values():
            for bias_type in result.get("bias_types", []):
                bias_type_counts[bias_type] += 1
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_hotels_analyzed": total_hotels,
            "hotels_with_bias": biased_hotels,
            "bias_detection_rate": f"{biased_hotels/total_hotels*100:.1f}%" if total_hotels > 0 else "0%",
            "bias_type_distribution": dict(bias_type_counts),
            "configuration": self.config,
            "detection_method": "keyword_based_no_embeddings"
        }
        
        # Save summary
        summary_path = self.output_dir / "bias-detection-summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary report: {summary_path}")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("BIAS DETECTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total hotels analyzed: {total_hotels}")
        logger.info(f"Hotels with bias: {biased_hotels}")
        logger.info(f"Bias detection rate: {summary['bias_detection_rate']}")
        logger.info("\nBias type distribution:")
        for bias_type, count in bias_type_counts.items():
            logger.info(f"  - {bias_type}: {count}")
        logger.info("="*70)
        
        return summary
    
    
    def run(self):
        """Run complete bias detection pipeline"""
        
        logger.info("="*70)
        logger.info("HOTEL BIAS DETECTION (Keyword-Based)")
        logger.info("="*70)
        
        # Load data
        logger.info("\n[1/4] Loading data...")
        reviews_df = self.load_reviews()
        responses_df = self.load_chatbot_responses()
        
        # Run bias detection
        logger.info(f"\n[2/4] Running bias detection on {len(responses_df)} hotels...")
        
        for idx, row in responses_df.iterrows():
            hotel_id = str(row['hotel_id'])
            response = row['answer']
            
            logger.info(f"\n--- Hotel {idx+1}/{len(responses_df)}: {hotel_id} ---")
            
            # Get reviews for this hotel
            hotel_reviews = reviews_df[reviews_df['hotel_id'] == int(hotel_id)]
            
            if hotel_reviews.empty:
                logger.warning(f"No reviews found for hotel {hotel_id}, skipping")
                continue
            
            # Run detection
            bias_result = self.detect_bias(hotel_id, response, hotel_reviews)
            
            # Store results
            self.results[hotel_id] = bias_result
            
            # Store detailed results for Parquet
            detailed_result = {
                'hotel_id': hotel_id,
                'response': response,
                'question': row.get('question', 'N/A'),
                'num_reviews': len(hotel_reviews),
                'avg_rating': float(hotel_reviews['overall_rating'].mean()),
                'bias_detected': bias_result['bias_detected'],
                'bias_types': ','.join(bias_result['bias_types']),
                'positive_reviews': bias_result['review_sentiments'].get('positive', 0),
                'neutral_reviews': bias_result['review_sentiments'].get('neutral', 0),
                'negative_reviews': bias_result['review_sentiments'].get('negative', 0),
                'response_positive_prob': bias_result['response_sentiment'].get('positive', 0),
                'response_neutral_prob': bias_result['response_sentiment'].get('neutral', 0),
                'response_negative_prob': bias_result['response_sentiment'].get('negative', 0),
            }
            self.detailed_results.append(detailed_result)
        
        # Save results
        logger.info("\n[3/4] Saving results...")
        self.save_results()
        
        # Generate summary
        logger.info("\n[4/4] Generating summary report...")
        summary = self.generate_summary_report()
        
        logger.info("\n✓ Bias detection complete!")
        
        return summary


def main():
    """Main execution"""
    
    # Configure paths
    REVIEWS_PATH = "processed_boston_reviews.csv"
    RESPONSES_PARQUET_PATH = "chatbot_responses.parquet"
    OUTPUT_DIR = "evaluation/results"
    
    # Check if files exist
    if not os.path.exists(REVIEWS_PATH):
        logger.error(f"Reviews file not found: {REVIEWS_PATH}")
        return
    
    if not os.path.exists(RESPONSES_PARQUET_PATH):
        logger.error(f"Chatbot responses file not found: {RESPONSES_PARQUET_PATH}")
        logger.info("Please run generate_chatbot_responses_xai.py first!")
        return
    
    # Run detection
    detector = HotelBiasDetection(
        reviews_path=REVIEWS_PATH,
        responses_parquet_path=RESPONSES_PARQUET_PATH,
        output_dir=OUTPUT_DIR
    )
    
    summary = detector.run()
    
    return summary


if __name__ == "__main__":
    try:
        summary = main()
    except Exception as e:
        logger.exception(f"Bias detection failed: {e}")
        raise