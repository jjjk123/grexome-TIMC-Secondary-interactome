#!/usr/bin/perl

# 25/03/2020
# NTM

# Takes 2 arguments: $inDir $outDir
# - $inDir must contain cohort TSVs as produced by extractCohorts.pl,
#   filtered and reordered by filterVariants.pl and reorderColumns.pl,
#   possibly with patientIDs;
# - $outDir doesn't exist, it will be created and filled with one TSV
#   per infile named $cohort.final.csv
#
# The outfiles are identical to the infiles except we remove lines where
# COUNT_$cohort_HV == 0 AND COUNT_$cohort_HET == 0 : previous steps
# kept such lines if they had COUNT_*_OTHERCAUSE_* >=1, this was
# necessary for extractSamples (for samples with a casual gene).

use strict;
use warnings;


(@ARGV == 2) || die "needs 2 args: an inDir and a non-existant outDir\n";
my ($inDir, $outDir) = @ARGV;
(-d $inDir) ||
    die "inDir $inDir doesn't exist or isn't a directory\n";
opendir(INDIR, $inDir) ||
    die "cannot opendir inDir $inDir\n";
(-e $outDir) && 
    die "found argument outDir $outDir but it already exists, remove it or choose another name.\n";
mkdir($outDir) || die "cannot mkdir outDir $outDir\n";


#########################################################

while (my $inFile = readdir(INDIR)) {
    ($inFile =~ /^\./) && next;
    my $cohort;
    if ($inFile =~ (/^(\w+)\.filtered\.\S+\.csv$/)) {
	$cohort = $1;
    }
    else {
	warn "W: cannot parse filename of inFile $inFile, skipping it\n";
	next;
    }

    open(INFILE, "$inDir/$inFile") ||
	die "E: cannot open infile $inDir/$inFile\n";

    my $outFile = "$cohort.final.csv" ;
    open(OUTFILE, "> $outDir/$outFile") ||
	die "E cannot open outfile $outDir/$outFile: $!\n";


    ###################################
    # header line
    my $header = <INFILE>;
    chomp($header);
    my @headers = split(/\t/,$header);

    # columns of COUNT_$cohort_HV and HET
    my ($hvCol,$hetCol) = (-1,-1);

    foreach my $hi (0..$#headers) {
	if ($headers[$hi] eq "COUNT_$cohort"."_HV") {
	    $hvCol = $hi;
	}
	elsif ($headers[$hi] eq "COUNT_$cohort"."_HET") {
	    $hetCol = $hi;
	}
    }
    # sanity check
    (($hvCol >= 0) && ($hetCol >= 0)) || 
	die "E: couldn't find one of HV/HET for $cohort\n";

    print OUTFILE "$header\n";

    # data lines
    while (my $line = <INFILE>) {
	chomp($line);
	my @fields = split(/\t/, $line, -1) ;
	if (($fields[$hvCol] == 0) && ($fields[$hetCol] == 0)) {
	    next;
	}
	else {
	    print OUTFILE "$line\n";
	}
    }
    close(INFILE);
    close(OUTFILE);
}

closedir(INDIR);