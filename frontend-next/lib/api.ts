const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface JobResponse {
  job_id: string;
  status: string;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "downloading" | "processing" | "complete" | "error";
  progress: number;
  message: string;
  output_path?: string;
  error?: string;
}

export interface ProcessJsonRequest {
  a_roll: {
    url: string;
    metadata?: string;
  };
  b_rolls: Array<{
    id: string;
    url: string;
    metadata?: string;
  }>;
}

export async function processFromJson(data: ProcessJsonRequest): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/api/process/json`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to start processing");
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${API_BASE_URL}/api/status/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get job status");
  }

  return response.json();
}

export async function downloadVideo(jobId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/api/download/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to download video");
  }

  return response.blob();
}
