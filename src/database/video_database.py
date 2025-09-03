"""
Centralized video database module for tracking all processed videos
Any script that processes videos should import and use this module to log entries
"""
import json
import os
from pathlib import Path
from datetime import datetime
import hashlib
import re
from typing import Optional, Dict, List, Any, Tuple


class VideoDatabase:
    """Manages a JSON database of processed videos"""
    
    def __init__(self, db_path: str = "video_database.json"):
        """Initialize the database"""
        self.db_path = Path(db_path)
        self.database = self._load_database()
    
    def _load_database(self) -> Dict[str, Any]:
        """Load existing database or create new one"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'generated_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_videos': 0,
                'statistics': {
                    'by_platform': {},
                    'by_category': {},
                    'by_source': {},  # Which script processed it
                    'total_characters': 0,
                    'total_words': 0,
                    'total_duration_seconds': 0,
                    'with_postprocessing': 0,
                    'with_asics': 0,
                    'processing_costs': {
                        'vast_ai_hours': 0,
                        'vast_ai_cost_usd': 0,
                        'openrouter_tokens': 0,
                        'openrouter_cost_usd': 0
                    }
                },
                'videos': [],
                'skipped_videos': []  # Track videos we've checked and skipped
            }
    
    def _save_database(self):
        """Save database to file"""
        self.database['last_updated'] = datetime.now().isoformat()
        with open(self.db_path, 'w') as f:
            json.dump(self.database, f, indent=2)
    
    @staticmethod
    def extract_video_id(filename: str) -> tuple[Optional[str], Optional[str]]:
        """Extract video ID and platform from filename"""
        # YouTube ID pattern (11 characters)
        youtube_pattern = r'[A-Za-z0-9_-]{11}'
        # X/Twitter ID pattern (long number)
        twitter_pattern = r'\d{16,20}'
        
        # Check for YouTube ID
        if match := re.search(rf'[_-]({youtube_pattern})(?:_transcript)?(?:\.json)?$', filename):
            return match.group(1), 'youtube'
        
        # Check for Twitter ID
        if match := re.search(rf'({twitter_pattern})', filename):
            return match.group(1), 'twitter'
        
        return None, None
    
    def add_video(
        self,
        transcript_path: Path,
        title: str,
        url: Optional[str] = None,
        platform: Optional[str] = None,
        category: Optional[str] = None,
        source_script: Optional[str] = None,
        processing_stats: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a new video to the database
        
        Args:
            transcript_path: Path to the transcript file
            title: Video title
            url: Video URL
            platform: Platform (youtube, twitter, etc)
            category: Category (bitcoin, gavin_mehl, etc)
            source_script: Name of script that processed this
            processing_stats: Stats like duration, cost, etc
            metadata: Additional metadata
            
        Returns:
            Unique ID of the added video
        """
        # Extract video ID if not provided
        if not platform:
            video_id, detected_platform = self.extract_video_id(str(transcript_path))
            platform = detected_platform or 'unknown'
        else:
            video_id, _ = self.extract_video_id(str(transcript_path))
        
        # Generate unique ID
        unique_id = f"{platform}_{video_id}" if video_id else f"{platform}_{datetime.now().timestamp()}"
        
        # Check if already exists
        if self.get_video(unique_id):
            print(f"Video already in database: {unique_id}")
            return unique_id
        
        # Create entry
        video_entry = {
            'id': unique_id,
            'video_id': video_id,
            'platform': platform,
            'title': title,
            'transcript_path': str(transcript_path),
            'processed_date': datetime.now().isoformat(),
            'source_script': source_script or 'unknown'
        }
        
        # Add URL
        if url:
            video_entry['url'] = url
        elif platform == 'youtube' and video_id:
            video_entry['url'] = f"https://www.youtube.com/watch?v={video_id}"
        elif platform == 'twitter' and video_id:
            video_entry['url'] = f"https://x.com/i/status/{video_id}"
        
        # Add category
        if category:
            video_entry['category'] = category
        else:
            # Auto-detect
            path_str = str(transcript_path).lower()
            if 'bitcoin' in path_str:
                video_entry['category'] = 'bitcoin'
            elif 'gavin' in path_str:
                video_entry['category'] = 'gavin_mehl'
            else:
                video_entry['category'] = 'general'
        
        # Add processing stats
        if processing_stats:
            video_entry['processing_stats'] = processing_stats
            
            # Update global stats
            if 'duration_seconds' in processing_stats:
                self.database['statistics']['total_duration_seconds'] += processing_stats['duration_seconds']
            if 'word_count' in processing_stats:
                self.database['statistics']['total_words'] += processing_stats['word_count']
            if 'char_count' in processing_stats:
                self.database['statistics']['total_characters'] += processing_stats['char_count']
            if 'vast_ai_cost' in processing_stats:
                if 'processing_costs' not in self.database['statistics']:
                    self.database['statistics']['processing_costs'] = {
                        'vast_ai_hours': 0,
                        'vast_ai_cost_usd': 0,
                        'openrouter_tokens': 0,
                        'openrouter_cost_usd': 0
                    }
                self.database['statistics']['processing_costs']['vast_ai_cost_usd'] += processing_stats['vast_ai_cost']
            if 'openrouter_cost' in processing_stats:
                if 'processing_costs' not in self.database['statistics']:
                    self.database['statistics']['processing_costs'] = {
                        'vast_ai_hours': 0,
                        'vast_ai_cost_usd': 0,
                        'openrouter_tokens': 0,
                        'openrouter_cost_usd': 0
                    }
                self.database['statistics']['processing_costs']['openrouter_cost_usd'] += processing_stats['openrouter_cost']
        
        # Add metadata
        if metadata:
            video_entry['metadata'] = metadata
        
        # Add to database
        self.database['videos'].insert(0, video_entry)  # Add to beginning
        self.database['total_videos'] += 1
        
        # Update statistics
        self.database['statistics']['by_platform'][platform] = \
            self.database['statistics']['by_platform'].get(platform, 0) + 1
        self.database['statistics']['by_category'][video_entry['category']] = \
            self.database['statistics']['by_category'].get(video_entry['category'], 0) + 1
        self.database['statistics']['by_source'][source_script or 'unknown'] = \
            self.database['statistics']['by_source'].get(source_script or 'unknown', 0) + 1
        
        # Save
        self._save_database()
        
        print(f"✓ Added to database: {title[:60]}...")
        return unique_id
    
    def update_video(self, video_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing video entry"""
        for video in self.database['videos']:
            if video['id'] == video_id:
                video.update(updates)
                video['last_modified'] = datetime.now().isoformat()
                self._save_database()
                return True
        return False
    
    def add_postprocessing(
        self,
        video_id: str,
        postprocessed_path: Path,
        clean_text_path: Optional[Path] = None,
        model: str = "claude-sonnet-4",
        cost: float = 0
    ) -> bool:
        """Add post-processing information to existing video"""
        for video in self.database['videos']:
            if video['id'] == video_id:
                video['postprocessed'] = True
                video['postprocessed_path'] = str(postprocessed_path)
                video['postprocess_model'] = model
                
                if clean_text_path:
                    video['clean_text_path'] = str(clean_text_path)
                
                if 'processing_stats' not in video:
                    video['processing_stats'] = {}
                video['processing_stats']['postprocess_cost'] = cost
                
                self.database['statistics']['with_postprocessing'] += 1
                if 'processing_costs' not in self.database['statistics']:
                    self.database['statistics']['processing_costs'] = {
                        'vast_ai_hours': 0,
                        'vast_ai_cost_usd': 0,
                        'openrouter_tokens': 0,
                        'openrouter_cost_usd': 0
                    }
                self.database['statistics']['processing_costs']['openrouter_cost_usd'] += cost
                
                self._save_database()
                return True
        return False
    
    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get a video entry by ID"""
        for video in self.database['videos']:
            if video['id'] == video_id:
                return video
        return None
    
    def get_videos_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all videos in a category"""
        return [v for v in self.database['videos'] if v.get('category') == category]
    
    def get_videos_by_platform(self, platform: str) -> List[Dict[str, Any]]:
        """Get all videos from a platform"""
        return [v for v in self.database['videos'] if v.get('platform') == platform]
    
    def get_recent_videos(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent videos"""
        return self.database['videos'][:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.database['statistics']
    
    def search_videos(self, query: str) -> List[Dict[str, Any]]:
        """Search videos by title or URL"""
        query_lower = query.lower()
        results = []
        for video in self.database['videos']:
            if (query_lower in video.get('title', '').lower() or 
                query_lower in video.get('url', '').lower() or
                query_lower in video.get('video_id', '').lower()):
                results.append(video)
        return results
    
    def add_skipped_video(
        self,
        video_id: str,
        title: str,
        url: str,
        reason: str,
        duration_seconds: Optional[int] = None,
        platform: str = 'youtube'
    ) -> bool:
        """
        Add a video to the skipped list
        
        Args:
            video_id: Video ID
            title: Video title
            url: Video URL
            reason: Why it was skipped (e.g., "too_short", "preview_clip")
            duration_seconds: Duration if known
            platform: Platform (youtube, twitter, etc)
            
        Returns:
            True if added, False if already exists
        """
        # Check if already in skipped list
        unique_id = f"{platform}_{video_id}"
        if self.is_video_skipped(video_id, platform):
            return False
        
        skipped_entry = {
            'id': unique_id,
            'video_id': video_id,
            'title': title,
            'url': url,
            'platform': platform,
            'reason': reason,
            'checked_date': datetime.now().isoformat()
        }
        
        if duration_seconds is not None:
            skipped_entry['duration_seconds'] = duration_seconds
            skipped_entry['duration_minutes'] = duration_seconds / 60
        
        # Add to skipped list
        if 'skipped_videos' not in self.database:
            self.database['skipped_videos'] = []
        
        self.database['skipped_videos'].append(skipped_entry)
        self._save_database()
        
        return True
    
    def is_video_skipped(self, video_id: str, platform: str = 'youtube') -> bool:
        """
        Check if a video is in the skipped list
        
        Args:
            video_id: Video ID to check
            platform: Platform (youtube, twitter, etc)
            
        Returns:
            True if video was previously skipped
        """
        if 'skipped_videos' not in self.database:
            return False
        
        unique_id = f"{platform}_{video_id}"
        for video in self.database['skipped_videos']:
            if video.get('id') == unique_id or video.get('video_id') == video_id:
                return True
        return False
    
    def is_video_known(self, video_id: str, platform: str = 'youtube') -> Tuple[bool, str]:
        """
        Check if a video is known (either processed or skipped)
        
        Args:
            video_id: Video ID to check
            platform: Platform (youtube, twitter, etc)
            
        Returns:
            Tuple of (is_known, status) where status is 'processed', 'skipped', or 'unknown'
        """
        unique_id = f"{platform}_{video_id}"
        
        # Check processed videos
        if self.get_video(unique_id):
            return True, 'processed'
        
        # Check skipped videos
        if self.is_video_skipped(video_id, platform):
            return True, 'skipped'
        
        return False, 'unknown'
    
    def export_summary(self) -> str:
        """Export a text summary of the database"""
        stats = self.database['statistics']
        summary = [
            "=" * 60,
            "VIDEO DATABASE SUMMARY",
            "=" * 60,
            f"Total Videos: {self.database['total_videos']}",
            f"Last Updated: {self.database.get('last_updated', 'N/A')}",
            "",
            "By Platform:",
        ]
        
        for platform, count in stats['by_platform'].items():
            summary.append(f"  - {platform}: {count}")
        
        summary.extend([
            "",
            "By Category:",
        ])
        
        for category, count in stats['by_category'].items():
            summary.append(f"  - {category}: {count}")
        
        summary.extend([
            "",
            "Processing Stats:",
            f"  - Total Words: {stats.get('total_words', 0):,}",
            f"  - Total Characters: {stats.get('total_characters', 0):,}",
            f"  - Total Duration: {stats.get('total_duration_seconds', 0) / 3600:.1f} hours",
            f"  - With Post-processing: {stats.get('with_postprocessing', 0)}",
        ])
        
        if 'processing_costs' in stats:
            costs = stats['processing_costs']
            summary.extend([
                "",
                "Processing Costs:",
                f"  - Vast.ai: ${costs.get('vast_ai_cost_usd', 0):.2f}",
                f"  - OpenRouter: ${costs.get('openrouter_cost_usd', 0):.2f}",
                f"  - Total: ${costs.get('vast_ai_cost_usd', 0) + costs.get('openrouter_cost_usd', 0):.2f}",
            ])
        
        summary.extend([
            "",
            "Recent Videos:",
        ])
        
        for video in self.get_recent_videos(5):
            summary.append(f"  - {video['title'][:50]}...")
            if 'url' in video:
                summary.append(f"    {video['url']}")
        
        return "\n".join(summary)


# Singleton instance for easy import
_db_instance = None

def get_database(db_path: str = "video_database.json") -> VideoDatabase:
    """Get or create the singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = VideoDatabase(db_path)
    return _db_instance


# Convenience functions for direct import
def log_video(transcript_path: Path, title: str, **kwargs) -> str:
    """Quick function to log a video to the database"""
    db = get_database()
    return db.add_video(transcript_path, title, **kwargs)


def log_postprocessing(video_id: str, postprocessed_path: Path, **kwargs) -> bool:
    """Quick function to log post-processing"""
    db = get_database()
    return db.add_postprocessing(video_id, postprocessed_path, **kwargs)