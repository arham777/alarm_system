import { User, AuthState, LoginCredentials } from '@/types/auth';

const AUTH_STORAGE_KEY = 'plant-health-auth';
const SESSION_DURATION = 24 * 60 * 60 * 1000; // 24 hours

// Hardcoded credentials for demo
const VALID_CREDENTIALS = {
  email: 'admin@gmail.com',
  password: 'admin123',
};

export class AuthService {
  static authenticate(credentials: LoginCredentials): boolean {
    return (
      credentials.email === VALID_CREDENTIALS.email &&
      credentials.password === VALID_CREDENTIALS.password
    );
  }

  static login(credentials: LoginCredentials): User | null {
    if (!this.authenticate(credentials)) {
      return null;
    }

    const user: User = {
      email: credentials.email,
      name: 'Plant Health Admin',
    };

    const authState: AuthState = {
      isAuthenticated: true,
      user,
      expiresAt: Date.now() + SESSION_DURATION,
    };

    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(authState));
    return user;
  }

  static logout(): void {
    localStorage.removeItem(AUTH_STORAGE_KEY);
  }

  static getCurrentUser(): User | null {
    try {
      const stored = localStorage.getItem(AUTH_STORAGE_KEY);
      if (!stored) return null;

      const authState: AuthState = JSON.parse(stored);
      
      // Check if session has expired
      if (authState.expiresAt && Date.now() > authState.expiresAt) {
        this.logout();
        return null;
      }

      return authState.user;
    } catch {
      return null;
    }
  }

  static isAuthenticated(): boolean {
    return this.getCurrentUser() !== null;
  }
}