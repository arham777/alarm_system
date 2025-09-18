export interface User {
  email: string;
  name: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  expiresAt: number | null;
}

export interface LoginCredentials {
  email: string;
  password: string;
}