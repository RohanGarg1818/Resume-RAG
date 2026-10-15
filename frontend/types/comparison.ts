export interface ComparisonRequest {
  job_requirements: string;
}

export interface RequirementMatch {
  requirement: string;
  met: boolean;
  evidence: string;
  confidence: string;
  candidate: string;
}

export interface RequirementComparison {
  requirement: string;
  candidate_a: RequirementMatch;
  candidate_b: RequirementMatch;
  better_match: string;
}

export interface CandidateScore {
  overall: number;
  required_skills: number;
  experience: number;
  projects: number;
  preferred_skills: number;
  education: number;
}

export interface CandidateAnalysis {
  name: string;
  score: CandidateScore;
  strengths: string[];
  gaps: string[];
  evidence: { text: string; section: string; relevance: number }[];
  unique_skills: string[];
  shared_strengths: string[];
}

export interface ComparisonResponse {
  recommendation: string;
  reasoning: string;
  score_a: CandidateScore;
  score_b: CandidateScore;
  requirement_comparisons: RequirementComparison[];
  candidate_a: CandidateAnalysis;
  candidate_b: CandidateAnalysis;
  missing_requirements: string[];
  ambiguous_requirements: string[];
  is_tie: boolean;
}
