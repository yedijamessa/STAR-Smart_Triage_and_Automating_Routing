export type TicketInput = {
  account: string;
  issue: string;
  contact?: string;
  project?: string;
  area?: string;
  input_category?: string;
  channel?: string;
  user_id?: string;
}

export type StreamPayload = {
  ticket_id?: string;
  intent?: string;
  entities?: { keywords?: string[] };
  sentiment?: string;
  category?: string;
  priority?: string;
  priority_level?: string;
  assigned_to?: string;
  auto_response?: string;
  resolution?: string;
  similar_tickets?: Array<{ full: string; solution?: string }>;
  resolution_similarity_score?: number;
  inferred_urgency?: string;
  inferred_urgency_code?: number;
}
