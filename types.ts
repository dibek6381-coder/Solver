export type View = 'design' | 'code' | 'training' | 'benchmarks';

export interface CodeFile {
  name: string;
  language: string;
  content: string;
  description: string;
}

export interface TrainingPhase {
  name: string;
  description: string;
  dataset: string;
  params: Record<string, string | number>;
}

export interface SystemModule {
  id: string;
  title: string;
  description: string;
  x: number;
  y: number;
}
