import { useState, useEffect } from 'react';
import { User, LoginCredentials } from '@/types/auth';
import { AuthService } from '@/lib/auth';

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const currentUser = AuthService.getCurrentUser();
    setUser(currentUser);
    setIsLoading(false);
  }, []);

  const login = (credentials: LoginCredentials) => {
    const user = AuthService.login(credentials);
    setUser(user);
    return user !== null;
  };

  const logout = () => {
    AuthService.logout();
    setUser(null);
  };

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    logout,
  };
}