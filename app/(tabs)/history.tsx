import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Calendar, TriangleAlert as AlertTriangle, CircleCheck as CheckCircle, Clock, Eye } from 'lucide-react-native';
import { useScans } from '@/hooks/useScans';

export default function HistoryScreen() {
  const { scans, isLoading } = useScans();
  const [selectedFilter, setSelectedFilter] = useState('all');

  const filters = [
    { id: 'all', label: 'All Scans', color: '#6B7280' },
    { id: 'benign', label: 'Looks Good', color: '#10B981' },
    { id: 'concerning', label: 'Needs Attention', color: '#EF4444' },
    { id: 'analyzing', label: 'Analyzing', color: '#F59E0B' },
  ];

  const filteredHistory = selectedFilter === 'all' 
    ? scans 
    : scans.filter(scan => scan.result === selectedFilter);

  const getRiskColor = (result: string) => {
    switch (result) {
      case 'benign': return '#10B981';
      case 'concerning': return '#EF4444';
      case 'analyzing': return '#F59E0B';
      default: return '#6B7280';
    }
  };

  const getRiskIcon = (result: string) => {
    switch (result) {
      case 'benign': return CheckCircle;
      case 'concerning': return AlertTriangle;
      case 'analyzing': return Clock;
      default: return CheckCircle;
    }
  };

  const getRiskLabel = (result: string) => {
    switch (result) {
      case 'benign': return 'Looks Good';
      case 'concerning': return 'See Doctor';
      case 'analyzing': return 'Analyzing...';
      default: return 'Unknown';
    }
  };

  const getResultDescription = (result: string) => {
    switch (result) {
      case 'benign': return 'Regular monitoring recommended';
      case 'concerning': return 'Professional evaluation recommended';
      case 'analyzing': return 'Analysis in progress...';
      default: return 'No description available';
    }
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Loading scan history...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Scan History</Text>
        <Text style={styles.subtitle}>Track your skin health over time</Text>
      </View>

      {/* Filters */}
      <View style={styles.filtersContainer}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filters}>
          {filters.map((filter) => (
            <TouchableOpacity
              key={filter.id}
              style={[
                styles.filterButton,
                selectedFilter === filter.id && { backgroundColor: filter.color },
              ]}
              onPress={() => setSelectedFilter(filter.id)}>
              <Text
                style={[
                  styles.filterText,
                  selectedFilter === filter.id && styles.filterTextActive,
                ]}>
                {filter.label}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* History List */}
      <ScrollView style={styles.historyList} showsVerticalScrollIndicator={false}>
        {filteredHistory.length === 0 ? (
          <View style={styles.emptyState}>
            <Calendar size={48} color="#9CA3AF" strokeWidth={2} />
            <Text style={styles.emptyTitle}>No scans found</Text>
            <Text style={styles.emptySubtitle}>
              {selectedFilter === 'all' 
                ? 'Start by taking your first skin scan' 
                : `No ${selectedFilter === 'benign' ? 'good results' : selectedFilter === 'concerning' ? 'concerning results' : 'analyzing scans'} in your history`}
            </Text>
          </View>
        ) : (
          filteredHistory.map((scan) => {
            const RiskIcon = getRiskIcon(scan.result);
            const riskColor = getRiskColor(scan.result);
            
            return (
              <View key={scan.id} style={styles.scanCard}>
                <View style={styles.scanHeader}>
                  <Image source={{ uri: scan.uri }} style={styles.scanImage} />
                  <View style={styles.scanInfo}>
                    <View style={styles.scanTitleRow}>
                      <Text style={styles.scanLocation}>{scan.location || 'Skin Scan'}</Text>
                      <View style={[styles.riskBadge, { backgroundColor: `${riskColor}15` }]}>
                        <RiskIcon size={12} color={riskColor} strokeWidth={2} />
                        <Text style={[styles.riskText, { color: riskColor }]}>
                          {getRiskLabel(scan.result)}
                        </Text>
                      </View>
                    </View>
                    <Text style={styles.scanDate}>{scan.dateScanned}</Text>
                    {scan.confidence && (
                      <Text style={styles.confidence}>Confidence: {scan.confidence}%</Text>
                    )}
                  </View>
                </View>
                
                <Text style={styles.scanNotes}>{getResultDescription(scan.result)}</Text>
                
                <TouchableOpacity style={styles.viewButton}>
                  <Eye size={16} color="#0066CC" strokeWidth={2} />
                  <Text style={styles.viewButtonText}>View Details</Text>
                </TouchableOpacity>
              </View>
            );
          })
        )}
      </ScrollView>

      {/* Summary Stats */}
      <View style={styles.statsContainer}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{scans.length}</Text>
          <Text style={styles.statLabel}>Total Scans</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={[styles.statValue, { color: '#10B981' }]}>
            {scans.filter(s => s.result === 'benign').length}
          </Text>
          <Text style={styles.statLabel}>Looks Good</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={[styles.statValue, { color: '#EF4444' }]}>
            {scans.filter(s => s.result === 'concerning').length}
          </Text>
          <Text style={styles.statLabel}>Needs Attention</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={[styles.statValue, { color: '#F59E0B' }]}>
            {scans.filter(s => s.result === 'analyzing').length}
          </Text>
          <Text style={styles.statLabel}>Analyzing</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  header: {
    padding: 24,
    paddingBottom: 16,
  },
  title: {
    fontSize: 28,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  filtersContainer: {
    paddingHorizontal: 24,
    marginBottom: 16,
  },
  filters: {
    flexDirection: 'row',
    gap: 12,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#E5E7EB',
  },
  filterText: {
    fontSize: 14,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
  },
  filterTextActive: {
    color: '#FFFFFF',
  },
  historyList: {
    flex: 1,
    paddingHorizontal: 24,
  },
  scanCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  scanHeader: {
    flexDirection: 'row',
    marginBottom: 12,
  },
  scanImage: {
    width: 60,
    height: 60,
    borderRadius: 12,
    marginRight: 16,
  },
  scanInfo: {
    flex: 1,
  },
  scanTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  scanLocation: {
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
  },
  riskBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  riskText: {
    fontSize: 12,
    fontFamily: 'Inter-SemiBold',
    marginLeft: 4,
  },
  scanDate: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    marginBottom: 2,
  },
  confidence: {
    fontSize: 12,
    fontFamily: 'Inter-Medium',
    color: '#9CA3AF',
  },
  scanNotes: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    lineHeight: 20,
    marginBottom: 12,
  },
  viewButton: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
  },
  viewButtonText: {
    fontSize: 14,
    fontFamily: 'Inter-SemiBold',
    color: '#0066CC',
    marginLeft: 6,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    textAlign: 'center',
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: '#FFFFFF',
    paddingVertical: 20,
    paddingHorizontal: 24,
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
    textAlign: 'center',
  },
});