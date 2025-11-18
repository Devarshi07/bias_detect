# generate_chatbot_responses_xai.py

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatbotResponseGenerator:
    """Generate review summaries using XAI (Grok)"""
    
    def __init__(self):
        """Initialize XAI client"""
        logger.info("Initializing XAI (Grok)...")
        
        xai_key = os.getenv("XAI_API_KEY")
        
        if not xai_key:
            logger.error("❌ XAI_API_KEY not found!")
            raise ValueError("XAI_API_KEY not set")
        
        if not xai_key.startswith("xai-"):
            logger.error(f"❌ Invalid XAI_API_KEY format: {xai_key[:10]}...")
            raise ValueError("Invalid XAI_API_KEY format")
        
        logger.info(f"✓ XAI_API_KEY found: {xai_key[:10]}...{xai_key[-5:]}")
        
        try:
            self.client = OpenAI(
                api_key=xai_key,
                base_url="https://api.x.ai/v1"
            )
            
            # Test connection
            logger.info("Testing XAI API connection...")
            test_response = self.client.chat.completions.create(
                model="grok-3",  # ← Updated model name
                messages=[{"role": "user", "content": "Say 'Connection successful'"}],
                max_tokens=20
            )
            logger.info(f"✓ XAI API connection successful!")
            logger.info(f"✓ Using XAI (grok-3)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize XAI: {e}")
            raise
    
    
    def generate_summary(self, reviews_df: pd.DataFrame, hotel_id: str) -> str:
        """Generate review summary using XAI"""
        
        hotel_reviews = reviews_df[reviews_df['hotel_id'] == hotel_id]
        
        if hotel_reviews.empty:
            return f"No reviews available for hotel {hotel_id}."
        
        # Prepare reviews text
        reviews_list = []
        for _, row in hotel_reviews.head(15).iterrows():
            review_text = f"Rating: {row['overall_rating']}/5\n"
            review_text += f"Review: {row['review_text']}\n"
            
            if 'service_rating' in row and pd.notna(row['service_rating']):
                review_text += f"Service: {row['service_rating']}/5, "
            if 'cleanliness_rating' in row and pd.notna(row['cleanliness_rating']):
                review_text += f"Cleanliness: {row['cleanliness_rating']}/5, "
            if 'location_rating' in row and pd.notna(row['location_rating']):
                review_text += f"Location: {row['location_rating']}/5\n"
            
            reviews_list.append(review_text)
        
        reviews_text = "\n---\n".join(reviews_list)
        
        num_reviews = len(hotel_reviews)
        avg_rating = hotel_reviews['overall_rating'].mean()
        
        prompt = f"""You are a helpful hotel review summarization assistant. Based on the guest reviews below, provide a balanced and accurate summary of what guests say about this hotel.

Hotel Statistics:
- Total reviews: {num_reviews}
- Average rating: {avg_rating:.1f}/5

Guest Reviews:
{reviews_text}

Instructions:
1. Provide a comprehensive summary (3-4 sentences)
2. Mention both positive and negative aspects that guests commonly discuss
3. Be balanced and fair - don't over-emphasize negative reviews if most are positive, and vice versa
4. Highlight specific aspects like service, cleanliness, location, rooms if mentioned
5. If there are very few reviews (less than 4), acknowledge the limited data

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model="grok-3",  # ← Updated model name
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"XAI error for hotel {hotel_id}: {e}")
            raise


def generate_responses(
    reviews_path: str,
    output_path: str = "chatbot_responses.parquet",
    num_hotels: int = 5
):
    """Generate chatbot responses using XAI"""
    
    logger.info("="*70)
    logger.info("GENERATING CHATBOT RESPONSES WITH XAI (GROK-3)")
    logger.info("="*70)
    
    # Initialize generator
    generator = ChatbotResponseGenerator()
    
    # Load reviews
    logger.info(f"\n[1/3] Loading reviews from: {reviews_path}")
    
    if not os.path.exists(reviews_path):
        logger.error(f"Reviews file not found: {reviews_path}")
        raise FileNotFoundError(f"File not found: {reviews_path}")
    
    reviews_df = pd.read_csv(reviews_path)
    logger.info(f"✓ Loaded {len(reviews_df)} reviews")
    logger.info(f"✓ Unique hotels: {reviews_df['hotel_id'].nunique()}")
    
    # Select hotels
    unique_hotels = reviews_df['hotel_id'].unique()[:num_hotels]
    logger.info(f"✓ Selected {len(unique_hotels)} hotels for processing")
    logger.info(f"  Hotel IDs: {list(unique_hotels)}")
    
    # Generate summaries
    logger.info(f"\n[2/3] Generating summaries with XAI (Grok-3)...")
    logger.info("This may take a minute...\n")
    
    responses = []
    
    for idx, hotel_id in enumerate(unique_hotels, 1):
        logger.info(f"{'='*70}")
        logger.info(f"Hotel {idx}/{len(unique_hotels)}: {hotel_id}")
        logger.info(f"{'='*70}")
        
        try:
            # Get hotel info
            hotel_reviews = reviews_df[reviews_df['hotel_id'] == hotel_id]
            num_reviews = len(hotel_reviews)
            avg_rating = hotel_reviews['overall_rating'].mean()
            
            logger.info(f"Reviews: {num_reviews} | Avg Rating: {avg_rating:.1f}/5")
            logger.info("Generating summary...")
            
            # Generate summary
            summary = generator.generate_summary(reviews_df, hotel_id)
            
            # Store response
            responses.append({
                'hotel_id': str(hotel_id),
                'question': 'What do guests say about this hotel?',
                'answer': summary,
                'timestamp': datetime.now().isoformat(),
                'num_reviews': num_reviews,
                'avg_rating': float(avg_rating),
                'generation_method': 'llm',
                'model': 'xai-grok-3'
            })
            
            logger.info("✓ Summary generated!\n")
            logger.info(f"Summary:\n{summary}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to generate summary for hotel {hotel_id}: {e}\n")
            continue
    
    if not responses:
        logger.error("No responses generated!")
        return None
    
    # Save responses
    logger.info(f"[3/3] Saving responses...")
    
    responses_df = pd.DataFrame(responses)
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    responses_df.to_parquet(output_path, index=False)
    logger.info(f"✓ Saved parquet: {output_path}")
    
    # Also save as CSV
    csv_path = output_path.replace('.parquet', '.csv')
    responses_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved CSV: {csv_path}")
    
    logger.info(f"\n{'='*70}")
    logger.info("GENERATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Hotels processed: {len(responses)}")
    logger.info(f"Output files:")
    logger.info(f"  - {output_path}")
    logger.info(f"  - {csv_path}")
    logger.info(f"{'='*70}\n")
    
    return responses_df


def main():
    """Main execution"""
    
    REVIEWS_PATH = "processed_boston_reviews.csv"
    OUTPUT_PATH = "response/chatbot_responses.parquet"
    NUM_HOTELS = 22
    
    try:
        responses_df = generate_responses(
            reviews_path=REVIEWS_PATH,
            output_path=OUTPUT_PATH,
            num_hotels=NUM_HOTELS
        )
        
        if responses_df is not None:
            logger.info("✓ All done!")
            logger.info(f"\nNext step: Run bias detection with:")
            logger.info(f"  python hotel_bias_detection.py")
        
        return responses_df
        
    except Exception as e:
        logger.exception(f"Failed to generate responses: {e}")
        raise


if __name__ == "__main__":
    responses_df = main()