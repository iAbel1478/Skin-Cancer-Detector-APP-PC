import { useState, useEffect, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface ScanResult {
  id: string;
  uri: string;
  result: 'benign' | 'concerning' | 'analyzing';
  confidence?: number;
  dateScanned: string;
  location?: string;
  notes?: string;
}

const SCANS_STORAGE_KEY = '@skinguard_scans';

// Global state management for real-time updates
let globalScans: ScanResult[] = [];
let listeners: Set<() => void> = new Set();

const notifyListeners = () => {
  console.log('Notifying listeners, current scans count:', globalScans.length);
  listeners.forEach(listener => {
    try {
      listener();
    } catch (error) {
      console.error('Error in listener:', error);
    }
  });
};

const updateGlobalScans = (newScans: ScanResult[]) => {
  console.log('Updating global scans from', globalScans.length, 'to', newScans.length);
  globalScans = [...newScans]; // Create a new array to ensure reference change
  notifyListeners();
};

export function useScans() {
  const [scans, setScans] = useState<ScanResult[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Subscribe to global state changes
  useEffect(() => {
    const listener = () => {
      console.log('Global scans updated, refreshing local state. New count:', globalScans.length);
      setScans([...globalScans]); // Create new array to trigger re-render
    };
    
    listeners.add(listener);
    console.log('Added listener, total listeners:', listeners.size);
    
    return () => {
      listeners.delete(listener);
      console.log('Removed listener, remaining listeners:', listeners.size);
    };
  }, []);

  // Load scans from storage on mount
  useEffect(() => {
    loadScans();
  }, []);

  const loadScans = async () => {
    try {
      console.log('Loading scans from storage...');
      setIsLoading(true);
      
      const storedScans = await AsyncStorage.getItem(SCANS_STORAGE_KEY);
      if (storedScans) {
        const parsedScans = JSON.parse(storedScans);
        console.log('Loaded scans from storage:', parsedScans.length);
        updateGlobalScans(parsedScans);
        setScans([...parsedScans]);
      } else {
        console.log('No stored scans found, starting with empty array');
        // Start with empty array instead of mock data
        updateGlobalScans([]);
        setScans([]);
      }
    } catch (error) {
      console.error('Error loading scans:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const saveScans = async (newScans: ScanResult[]) => {
    try {
      console.log('Saving scans to storage:', newScans.length);
      const scansToSave = [...newScans]; // Ensure we have a clean array
      await AsyncStorage.setItem(SCANS_STORAGE_KEY, JSON.stringify(scansToSave));
      
      // Update global state and notify all listeners
      updateGlobalScans(scansToSave);
      console.log('Scans saved and global state updated successfully');
      
      return scansToSave;
    } catch (error) {
      console.error('Error saving scans:', error);
      throw error;
    }
  };

  const addScan = async (scanData: Omit<ScanResult, 'id' | 'dateScanned'>) => {
    try {
      console.log('Adding new scan:', scanData);
      const newScan: ScanResult = {
        ...scanData,
        id: Date.now().toString(),
        dateScanned: new Date().toISOString().split('T')[0],
      };

      // Get the most current scans from global state
      const currentScans = [...globalScans];
      const updatedScans = [newScan, ...currentScans];
      await saveScans(updatedScans);

      // Simulate AI analysis
      if (newScan.result === 'analyzing') {
        console.log('Starting AI analysis simulation for scan:', newScan.id);
        setTimeout(async () => {
          try {
            const analyzedScan = {
              ...newScan,
              result: Math.random() > 0.7 ? 'concerning' : 'benign' as 'benign' | 'concerning',
              confidence: Math.floor(Math.random() * 20) + 80,
            };
            
            console.log('Analysis complete for scan:', analyzedScan.id, 'Result:', analyzedScan.result);
            
            // Update the scan with results
            const latestScans = [...globalScans];
            const updatedScansWithResult = latestScans.map((scan: ScanResult) => 
              scan.id === newScan.id ? analyzedScan : scan
            );
            await saveScans(updatedScansWithResult);
          } catch (error) {
            console.error('Error updating scan result:', error);
          }
        }, 3000);
      }

      return newScan;
    } catch (error) {
      console.error('Error adding scan:', error);
      throw error;
    }
  };

  const deleteScan = async (scanId: string) => {
    try {
      console.log('=== SIMPLE DELETE SCAN START ===');
      console.log('Deleting scan with ID:', scanId);
      
      // Get current scans from storage
      const storedScans = await AsyncStorage.getItem(SCANS_STORAGE_KEY);
      let currentScans = [];
      
      if (storedScans) {
        currentScans = JSON.parse(storedScans);
      }
      
      console.log('Current scans before delete:', currentScans.length);
      
      // Remove the scan
      const updatedScans = currentScans.filter(scan => scan.id !== scanId);
      
      console.log('Scans after delete:', updatedScans.length);
      
      // Save back to storage
      await AsyncStorage.setItem(SCANS_STORAGE_KEY, JSON.stringify(updatedScans));
      
      // Update local state
      setScans(updatedScans);
      
      console.log('=== SIMPLE DELETE SCAN COMPLETE ===');
      
      return true;
    } catch (error) {
      console.error('=== SIMPLE DELETE SCAN ERROR ===');
      console.error('Error deleting scan:', error);
      throw error;
    }
  };

  const updateScan = async (scanId: string, updates: Partial<ScanResult>) => {
    try {
      console.log('Updating scan:', scanId, updates);
      
      // Use current global scans
      const currentScans = [...globalScans];
      const updatedScans = currentScans.map(scan => 
        scan.id === scanId ? { ...scan, ...updates } : scan
      );
      await saveScans(updatedScans);
    } catch (error) {
      console.error('Error updating scan:', error);
      throw error;
    }
  };

  const refreshScans = useCallback(async () => {
    console.log('Refreshing scans...');
    await loadScans();
  }, []);

  const clearAllScans = useCallback(async () => {
    try {
      console.log('Clearing all scans...');
      await AsyncStorage.removeItem(SCANS_STORAGE_KEY);
      updateGlobalScans([]);
      setScans([]);
      console.log('All scans cleared successfully');
    } catch (error) {
      console.error('Error clearing scans:', error);
      throw error;
    }
  }, []);

  // Debug logging
  useEffect(() => {
    console.log('useScans state updated - scans count:', scans.length);
  }, [scans]);

  return {
    scans,
    isLoading,
    addScan,
    deleteScan,
    updateScan,
    refreshScans,
    clearAllScans,
  };
}